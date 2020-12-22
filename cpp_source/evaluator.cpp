#include <Eigen/Core>
#include <Eigen/Sparse>
#include <future>
#include <iostream>
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

  EvaluatorCore(const SparseMatrix &X)
      : X_(X), n_users(X.rows()), n_items(X.cols()) {
    X_.makeCompressed();
  }

  inline Metrics get_metrics(const Eigen::Ref<DenseMatrix> &scores,
                             size_t cutoff, size_t offset, size_t n_thread,
                             bool recall_with_cutoff = false) {
    Metrics overall(n_items);
    check_arg(n_users > offset, "got offset >= n_users");
    check_arg(cutoff > 0, "cutoff must be strictly greather than 0.");
    check_arg(cutoff <= n_items, "cutoff must not exeeed the number of items.");

    std::vector<std::vector<int>> uinds(n_thread);
    for (int u = 0; u < scores.rows(); u++) {
      uinds[u % n_thread].push_back(u);
    }
    std::vector<std::future<Metrics>> workers;
    for (size_t th = 0; th < n_thread; th++) {
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

    Metrics metrics(n_items);

    const Real *buffer = scores.data();
    std::unordered_set<StorageIndex> hit_item;
    std::vector<StorageIndex> index(n_items);
    std::vector<StorageIndex> recommendation(cutoff);
    std::vector<StorageIndex> intersection(cutoff);
    std::vector<double> dcg_discount(cutoff);
    hit_item.reserve(cutoff);
    for (size_t i = 0; i < cutoff; i++) {
      dcg_discount[i] = 1 / std::log2(2 + i);
    }
    for (size_t _ = 0; _ < n_items; _++) {
      index[_] = _;
    }
    for (int u : user_set) {
      int u_orig = u + offset;
      metrics.total_user += 1;
      int begin_ptr = u * n_items;
      std::sort(index.begin(), index.end(),
                [&buffer, &begin_ptr](int i1, int i2) {
                  return buffer[begin_ptr + i1] > buffer[begin_ptr + i2];
                });
      for (size_t i = 0; i < cutoff; i++) {
        metrics.item_cnt(index[i]) += 1;
      }
      size_t n_gt = X_.outerIndexPtr()[u_orig + 1] - X_.outerIndexPtr()[u_orig];
      if (n_gt == 0) {
        continue;
      }
      metrics.valid_user += 1;

      hit_item.clear();
      if (recall_with_cutoff) {
        n_gt = std::min(n_gt, cutoff);
      }
      std::copy(index.begin(), index.begin() + cutoff, recommendation.begin());
      std::sort(index.begin(), index.begin() + cutoff);
      const StorageIndex *gb_begin =
          X_.innerIndexPtr() + X_.outerIndexPtr()[u_orig];
      const StorageIndex *gb_end =
          X_.innerIndexPtr() + X_.outerIndexPtr()[u_orig + 1];
      auto it_end =
          std::set_intersection(gb_begin, gb_end, index.begin(),
                                index.begin() + cutoff, intersection.begin());
      size_t n_hit = std::distance(intersection.begin(), it_end);
      for (auto iter = intersection.begin(); iter != it_end; iter++) {
        hit_item.insert(*iter);
      }
      if (n_hit > 0) {
        metrics.hit += 1;
      }
      metrics.precision += n_hit / static_cast<double>(cutoff);
      metrics.recall += n_hit / static_cast<double>(n_gt);
      double dcg = 0;
      double idcg =
          std::accumulate(dcg_discount.begin(),
                          dcg_discount.begin() + std::min(n_gt, cutoff), 0.);
      double cum_hit = 0;
      double average_precision = 0;
      for (size_t i = 0; i < cutoff; i++) {
        if (hit_item.find(recommendation[i]) != hit_item.end()) {
          dcg += dcg_discount[i];
          cum_hit += 1;
          average_precision += (cum_hit / (i + 1));
        }
      }
      metrics.ndcg += (dcg / idcg);
      metrics.map += average_precision / n_gt;
    }
    return metrics;
  }

  SparseMatrix X_;
  const size_t n_users;
  const size_t n_items;
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
      .def(py::init<const typename EvaluatorCore::SparseMatrix &>(),
           py::arg("grount_truth"))
      .def("get_metrics", &EvaluatorCore::get_metrics, py::arg("score_array"),
           py::arg("cutoff"), py::arg("offset"), py::arg("n_thread"),
           py::arg("recall_with_cutoff") = false)
      .def("get_ground_truth", &EvaluatorCore::get_ground_truth);
}
