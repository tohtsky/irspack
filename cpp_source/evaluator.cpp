#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <iostream>
#include <string>
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

  template <typename ScoreFloatType>
  using DenseMatrix = Eigen::Matrix<ScoreFloatType, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>;

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

  inline void cache_X_map(size_t n_threads) {
    check_arg(n_threads > 0, "n_threads must be strictly positive.");
    if (!this->X_as_set.empty()) {
      return;
    }

    const int64_t n_rows = this->X_.rows();
    this->X_as_set.resize(this->X_.rows());
    std::atomic<std::int64_t> cursor(0);
    std::vector<std::future<void>> workers;
    for (size_t th = 0; th < n_threads; th++) {
      workers.emplace_back(std::async([this, &cursor, n_rows]() {
        while (true) {
          int64_t current = cursor.fetch_add(1);
          if (current >= n_rows) {
            break;
          }
          auto &target = this->X_as_set[current];
          for (SparseMatrix::InnerIterator iter(this->X_, current); iter;
               ++iter) {
            target.insert(iter.col());
          }
        }
      }));
    }
  }

  template <typename ScoreFloatType>
  inline Metrics
  get_metrics(const Eigen::Ref<DenseMatrix<ScoreFloatType>> &scores,
              size_t cutoff, size_t offset, size_t n_threads,
              bool recall_with_cutoff = false) {
    this->cache_X_map(n_threads);
    Metrics overall(n_items);
    check_arg(n_threads > 0, "n_threads == 0");
    check_arg(n_users > offset, "got offset >= n_users");
    check_arg(static_cast<size_t>(offset + scores.rows()) <= n_users,
              "offset + scores.shape[0] exceeds n_users");
    check_arg(cutoff > 0, "cutoff must be strictly greather than 0.");
    check_arg(cutoff <= n_items, "cutoff must not exeeed the number of items.");
    std::atomic<int64_t> current_index(0);

    std::vector<std::future<Metrics>> workers;
    for (size_t th = 0; th < n_threads; th++) {
      workers.emplace_back(std::async(
          std::launch::async, [this, &current_index, &scores, cutoff,
                               offset, recall_with_cutoff]() {
            return this->get_metrics_local(scores, current_index, cutoff,
                                           offset, recall_with_cutoff);
          }));
    }
    for (auto &metric : workers) {
      overall.merge(metric.get());
    }
    return overall;
  }

  inline SparseMatrix get_ground_truth() const { return this->X_; }
  inline std::vector<std::vector<size_t>> get_recommendable_items() const {
    return this->recommendable_items;
  }

private:
  template <typename ScoreFloatType>
  inline Metrics
  get_metrics_local(const Eigen::Ref<DenseMatrix<ScoreFloatType>> &scores,
                    std::atomic<int64_t> &current_index, size_t cutoff,
                    size_t offset, bool recall_with_cutoff = false) const {
    using StorageIndex = typename SparseMatrix::StorageIndex;

    Metrics metrics_local(n_items);

    const ScoreFloatType *score_begin = scores.data();
    std::vector<std::pair<ScoreFloatType, StorageIndex>> score_and_index;
    score_and_index.reserve(n_items);
    std::vector<StorageIndex> intersection(cutoff);
    std::vector<double> dcg_discount(cutoff);
    for (size_t i = 0; i < cutoff; i++) {
      dcg_discount[i] = 1 / std::log2(2 + i);
    }

    size_t n_recommendable_items = std::min(cutoff, n_items);
    while (true) {
      auto u = current_index.fetch_add(1);
      if (u >= scores.rows()) {
        break;
      }
      auto buffer = score_begin + n_items * u;
      int u_orig = u + offset;

      const auto &gt_indices = this->X_as_set.at(u_orig);
      metrics_local.total_user += 1;
      score_and_index.clear();

      size_t n_gt = 0;
      const auto n_items_signed = static_cast<StorageIndex>(n_items);
      if (this->recommendable_items.empty()) {
        auto score_loc = buffer;
        for (StorageIndex _ = 0; _ < n_items_signed; _++) {
          score_and_index.emplace_back(-*(score_loc++), _);
        }
        n_gt = gt_indices.size();
      } else if (this->recommendable_items.size() == 1u) {
        const auto &rec_items_global = recommendable_items[0];
        for (auto i : rec_items_global) {
          score_and_index.emplace_back(-buffer[i], i);
          if (gt_indices.find(i) != gt_indices.cend()) {
            n_gt++;
          }
        }
        n_recommendable_items = std::min(cutoff, recommendable_items[0].size());
      } else {
        const auto &item_local = this->recommendable_items[u_orig];
        for (auto i : item_local) {
          score_and_index.emplace_back(-buffer[i], i);
          if (gt_indices.find(i) != gt_indices.cend()) {
            n_gt++;
          }
        }
        n_recommendable_items = std::min(cutoff, item_local.size());
      }

      if ((n_gt == 0) || (n_recommendable_items == 0)) {
        continue;
      }

      std::partial_sort(score_and_index.begin(),
                        score_and_index.begin() + n_recommendable_items,
                        score_and_index.end());

      metrics_local.valid_user += 1;
      double dcg = 0;
      double idcg = std::accumulate(
          dcg_discount.begin(),
          dcg_discount.begin() + std::min(n_gt, n_recommendable_items), 0.);
      double average_precision = 0;

      size_t cum_hit = 0;
      for (size_t i = 0; i < n_recommendable_items; i++) {
        auto rec_index = score_and_index[i].second;
        metrics_local.item_cnt(rec_index) += 1;
        if (gt_indices.find(rec_index) == gt_indices.cend()) {
        } else {
          dcg += dcg_discount[i];
          cum_hit++;
          average_precision += (static_cast<Real>(cum_hit) / (i + 1));
        }
      }
      if (cum_hit > 0) {
        metrics_local.hit += 1;
      }
      metrics_local.precision +=
          cum_hit / static_cast<double>(n_recommendable_items);
      metrics_local.recall +=
          cum_hit / static_cast<double>(recall_with_cutoff
                                            ? (n_gt > cutoff ? cutoff : n_gt)
                                            : n_gt);
      metrics_local.ndcg += (dcg / idcg);
      metrics_local.map += average_precision / n_gt;
    }
    return metrics_local;
  }

  SparseMatrix X_;
  const size_t n_users;
  const size_t n_items;
  std::vector<std::vector<size_t>> recommendable_items;
  std::vector<std::unordered_set<int64_t>> X_as_set;
}; // namespace irspack
} // namespace irspack

namespace py = pybind11;
using namespace irspack;

PYBIND11_MODULE(_core, m) {
  py::class_<Metrics>(m, "Metrics")
      .def(py::init<size_t>())
      .def("merge",
           [](Metrics &this_, const Metrics &other) { this_.merge(other); })
      .def("as_dict", &Metrics::as_dict);

  py::class_<EvaluatorCore>(m, "EvaluatorCore")
      .def(py::init<const typename EvaluatorCore::SparseMatrix &,
                    const std::vector<std::vector<size_t>> &>(),
           py::arg("grount_truth"), py::arg("recommendable"))
      .def("get_metrics_f64",
           static_cast<Metrics (EvaluatorCore::*)(
               const Eigen::Ref<EvaluatorCore::DenseMatrix<double>> &, size_t,
               size_t, size_t, bool)>(&EvaluatorCore::get_metrics<double>),
           py::arg("score_array"), py::arg("cutoff"), py::arg("offset"),
           py::arg("n_threads"), py::arg("recall_with_cutoff") = false)
      .def("get_metrics_f32",
           static_cast<Metrics (EvaluatorCore::*)(
               const Eigen::Ref<EvaluatorCore::DenseMatrix<float>> &, size_t,
               size_t, size_t, bool)>(&EvaluatorCore::get_metrics<float>),
           py::arg("score_array"), py::arg("cutoff"), py::arg("offset"),
           py::arg("n_threads"), py::arg("recall_with_cutoff") = false)
      .def("get_ground_truth", &EvaluatorCore::get_ground_truth)
      .def("cache_X_as_set", &EvaluatorCore::cache_X_map)
      .def(py::pickle(
          [](const EvaluatorCore &evaluator) {
            return py::make_tuple(evaluator.get_ground_truth(),
                                  evaluator.get_recommendable_items());
          },
          [](py::tuple t) {
            if (t.size() != 2)
              throw std::runtime_error("invalid state");
            return EvaluatorCore(t[0].cast<EvaluatorCore::SparseMatrix>(),
                                 t[1].cast<std::vector<std::vector<size_t>>>());
          }));

  ;
}
