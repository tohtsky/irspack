#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "argcheck.hpp"

namespace irspack {

using CountVector = Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>;
struct Metrics {
  // This isn't necessary, but MSVC complains Metric is not default
  // constructible.
  inline Metrics() : Metrics(0) {}
  inline Metrics(size_t n_item) : n_item(n_item), item_cnt(n_item) {
    item_cnt.array() = 0;
  }

  inline void merge(const Metrics &other) {
    hit += other.hit;
    recall += other.recall;
    ndcg += other.ndcg;
    total_user += other.total_user;
    valid_user += other.valid_user;
    item_cnt += other.item_cnt;
    precision += other.precision;
    map += other.map;
  }

  std::map<std::string, double> as_dict() const {
    std::map<std::string, double> result;
    CountVector count_local(item_cnt);
    double total_item = item_cnt.sum();
    std::sort(count_local.data(), count_local.data() + item_cnt.rows());
    double apperere_item = 0;
    double entropy = 0;
    double gini = 0;

    for (int i = 0; i < item_cnt.rows(); i++) {
      int64_t cnt = count_local(i);
      if (cnt == 0) {
        continue;
      }
      double p = cnt / total_item;
      apperere_item++;
      entropy += -std::log(p) * p;
      gini += (2 * i - item_cnt.rows() + 1) * cnt;
    }
    gini /= (item_cnt.rows() * total_item);

    size_t denominator = valid_user > 0u ? valid_user : 1;
    result["total_user"] = total_user;
    result["valid_user"] = valid_user;
    result["n_items"] = n_item;
    result["hit"] = hit / denominator;
    result["ndcg"] = ndcg / denominator;
    result["recall"] = recall / denominator;
    result["map"] = map / denominator;
    result["precision"] = precision / denominator;
    result["appeared_item"] = apperere_item;
    result["entropy"] = entropy;
    result["gini_index"] = gini;
    return result;
  }

  size_t valid_user = 0;
  size_t total_user = 0;
  double hit = 0;
  double recall = 0;
  double ndcg = 0;
  double precision = 0;
  double map = 0;
  size_t n_item;
  CountVector item_cnt;
};

struct EvaluatorCore {
  using Real = double;
  using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
  using DenseMatrix =
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

  EvaluatorCore(const SparseMatrix &X,
                const std::vector<std::vector<size_t>> &recommendable_items)
      : X_(X), n_users(X.rows()), n_items(X.cols()),
        recommendable_items(recommendable_items) {
    check_arg(recommendable_items.empty() ||
                  (recommendable_items.size() == 1) ||
                  static_cast<size_t>(X.rows()) == recommendable_items.size(),
              "recommendable.size.() must be in {0, 1, ground_truth.size()}");
    X_.makeCompressed();
    // sort & bound check
    for (auto &urec : this->recommendable_items) {
      std::sort(urec.begin(), urec.end());
      if (!urec.empty()) {
        check_arg(urec.back() < static_cast<size_t>(X_.cols()),
                  "recommendable items contain a index >= n_items.");
        auto prev = urec[0];
        for (size_t i = 1; i < urec.size(); i++) {
          auto current = urec[i];
          check_arg(current > prev, "duplicate recommendable items.");
          prev = current;
        }
      }
    }
  }

  inline Metrics get_metrics(const Eigen::Ref<DenseMatrix> &scores,
                             size_t cutoff, size_t offset, size_t n_threads,
                             bool recall_with_cutoff = false) {
    Metrics overall(n_items);
    check_arg(n_threads > 0, "n_threads == 0");
    check_arg(n_users > offset, "got offset >= n_users");
    check_arg(cutoff > 0, "cutoff must be strictly greather than 0.");
    check_arg(cutoff <= n_items, "cutoff must not exeeed the number of items.");

    std::vector<std::vector<int>> uinds(n_threads);
    for (int u = 0; u < scores.rows(); u++) {
      uinds[u % n_threads].push_back(u);
    }
    std::vector<std::future<Metrics>> workers;
    for (size_t th = 0; th < n_threads; th++) {
      workers.emplace_back(
          std::async(std::launch::async, [th, this, &scores, &uinds, cutoff,
                                          offset, recall_with_cutoff]() {
            return this->get_metrics_local(scores, uinds[th], cutoff, offset,
                                           recall_with_cutoff);
          }));
    }
    for (auto &metric : workers) {
      overall.merge(metric.get());
    }
    return overall;
  }

  inline SparseMatrix get_ground_truth() const { return this->X_; }

private:
  inline Metrics get_metrics_local(const Eigen::Ref<DenseMatrix> &scores,
                                   const std::vector<int> &user_set,
                                   size_t cutoff, size_t offset,
                                   bool recall_with_cutoff = false) const {
    using StorageIndex = typename SparseMatrix::StorageIndex;

    Metrics metrics_local(n_items);

    const Real *buffer = scores.data();
    std::unordered_set<StorageIndex> hit_item;
    std::vector<StorageIndex> index;
    index.reserve(n_items);
    std::vector<StorageIndex> recommendable_ground_truths(n_items);
    std::vector<StorageIndex> recommendation(cutoff);
    std::vector<StorageIndex> intersection(cutoff);
    std::vector<double> dcg_discount(cutoff);
    hit_item.reserve(cutoff);
    for (size_t i = 0; i < cutoff; i++) {
      dcg_discount[i] = 1 / std::log2(2 + i);
    }

    size_t n_recommendable_items = std::min(cutoff, n_items);
    for (int u : user_set) {
      int u_orig = u + offset;
      metrics_local.total_user += 1;
      int begin_ptr = u * n_items;
      const StorageIndex *gb_begin =
          X_.innerIndexPtr() + X_.outerIndexPtr()[u_orig];
      const StorageIndex *gb_end =
          X_.innerIndexPtr() + X_.outerIndexPtr()[u_orig + 1];

      recommendation.clear();
      index.clear();
      if (this->recommendable_items.empty()) {
        for (size_t _ = 0; _ < n_items; _++) {
          index.push_back(_);
        }
      } else if (this->recommendable_items.size() == 1u) {
        std::copy(recommendable_items[0].begin(), recommendable_items[0].end(),
                  std::back_inserter(index));
        n_recommendable_items = std::min(cutoff, recommendable_items[0].size());
      } else {
        auto &item_local = this->recommendable_items[u_orig];
        std::copy(item_local.begin(), item_local.end(),
                  std::back_inserter(index));
        n_recommendable_items = std::min(cutoff, item_local.size());
      }

      recommendable_ground_truths.clear();
      std::set_intersection(index.begin(), index.end(), gb_begin, gb_end,
                            std::back_inserter(recommendable_ground_truths));
      size_t n_gt = recommendable_ground_truths.size();
      if ((n_gt == 0) || (n_recommendable_items == 0)) {
        continue;
      }
      std::sort(index.begin(), index.end(),
                [&buffer, &begin_ptr](int i1, int i2) {
                  return buffer[begin_ptr + i1] > buffer[begin_ptr + i2];
                });
      for (size_t i = 0; i < n_recommendable_items; i++) {
        metrics_local.item_cnt(index[i]) += 1;
      }
      metrics_local.valid_user += 1;

      hit_item.clear();
      std::copy(index.begin(), index.begin() + n_recommendable_items,
                std::back_inserter(recommendation));
      std::sort(index.begin(), index.begin() + n_recommendable_items);
      auto it_end = std::set_intersection(gb_begin, gb_end, index.begin(),
                                          index.begin() + n_recommendable_items,
                                          intersection.begin());
      size_t n_hit = std::distance(intersection.begin(), it_end);
      for (auto iter = intersection.begin(); iter != it_end; iter++) {
        hit_item.insert(*iter);
      }
      if (n_hit > 0) {
        metrics_local.hit += 1;
      }
      metrics_local.precision +=
          n_hit / static_cast<double>(n_recommendable_items);
      metrics_local.recall += n_hit / static_cast<double>(n_gt);
      double dcg = 0;
      double idcg = std::accumulate(
          dcg_discount.begin(),
          dcg_discount.begin() + std::min(n_gt, n_recommendable_items), 0.);
      double cum_hit = 0;
      double average_precision = 0;
      for (size_t i = 0; i < n_recommendable_items; i++) {
        if (hit_item.find(recommendation[i]) != hit_item.end()) {
          dcg += dcg_discount[i];
          cum_hit += 1;
          average_precision += (cum_hit / (i + 1));
        }
      }

      metrics_local.ndcg += (dcg / idcg);
      metrics_local.map += average_precision / n_gt;
    }
    return metrics_local;
  }

  SparseMatrix X_;
  const size_t n_users;
  const size_t n_items;
  std::vector<std::vector<size_t>> recommendable_items;
};
} // namespace irspack

namespace py = pybind11;
using namespace irspack;

PYBIND11_MODULE(_evaluator, m) {
  py::class_<Metrics>(m, "Metrics")
      .def(py::init<size_t>())
      .def("merge",
           [](Metrics &this_, const Metrics &other) { this_.merge(other); })
      .def("as_dict", &Metrics::as_dict);

  py::class_<EvaluatorCore>(m, "EvaluatorCore")
      .def(py::init<const typename EvaluatorCore::SparseMatrix &,
                    const std::vector<std::vector<size_t>> &>(),
           py::arg("grount_truth"), py::arg("recommendable"))
      .def("get_metrics", &EvaluatorCore::get_metrics, py::arg("score_array"),
           py::arg("cutoff"), py::arg("offset"), py::arg("n_threads"),
           py::arg("recall_with_cutoff") = false)
      .def("get_ground_truth", &EvaluatorCore::get_ground_truth);
}
