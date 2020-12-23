#include <future>
#include <map>
#include <mutex>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <thread>
#include <vector>

using namespace std;

using MatrixEntry = float;
using CSRMatrix = Eigen::SparseMatrix<MatrixEntry, Eigen::RowMajor>;
using ReturnValue = Eigen::SparseMatrix<int32_t, Eigen::RowMajor>;

static uniform_real_distribution<float> udist_(0, 1);
namespace py = pybind11;

namespace irspack {

struct RandomWalkGenerator {
  inline RandomWalkGenerator(CSRMatrix X)
      : user_item(X), item_user(X.transpose()), n_item(X.cols()),
        n_user(X.rows()) {
    user_item.makeCompressed();
    item_user.makeCompressed();
  }

protected:
  inline size_t step_i_to_u(size_t start_ind, std::mt19937 &rns) const {
    size_t start_index = item_user.outerIndexPtr()[start_ind];
    size_t ep = item_user.outerIndexPtr()[start_ind + 1];
    size_t nnz = ep - start_index;
    size_t indptr = start_index + floor(nnz * udist_(rns));
    return item_user.innerIndexPtr()[indptr];
  }

  inline size_t step_u_to_i(size_t start_ind, std::mt19937 &rns) const {
    size_t start_index = user_item.outerIndexPtr()[start_ind];
    size_t ep = user_item.outerIndexPtr()[start_ind + 1];
    size_t nnz = ep - start_index;
    size_t indptr = start_index + floor(nnz * udist_(rns));
    return user_item.innerIndexPtr()[indptr];
  }

public:
  ReturnValue run_with_restart(float decay, size_t cutoff, size_t n_count,
                               size_t n_worker, int random_seed) const {
    using Triplet = Eigen::Triplet<int32_t>;
    std::vector<std::future<std::vector<Triplet>>> futures;
    for (size_t thread_id = 0; thread_id < n_worker; thread_id++) {
      futures.push_back(std::async(
          [this, decay, cutoff, thread_id, n_count, n_worker, random_seed]() {
            std::mt19937 rns(random_seed + thread_id);
            vector<Triplet> d;
            for (size_t i = thread_id; i < this->n_item; i += n_worker) {
              auto counts =
                  this->_run_item_walk_restart(decay, cutoff, i, n_count, rns);
              for (auto &iter : counts) {
                d.emplace_back(i, iter.first, iter.second);
              }
            }
            return d;
          }));
    }
    vector<Triplet> d;
    for (size_t thread_id = 0; thread_id < n_worker; thread_id++) {
      vector<Triplet> _ = futures[thread_id].get();
      d.insert(d.end(), _.begin(), _.end());
    }

    ReturnValue result(n_item, n_item);
    result.setFromTriplets(d.begin(), d.end());
    result.makeCompressed();
    return result;
  }

protected:
  inline map<size_t, int> _run_item_walk_fixed_step(size_t item_start_index,
                                                    size_t n_step,
                                                    size_t n_count,
                                                    std::mt19937 &rns) const {

    map<size_t, int> count;
    auto current_loc = item_start_index;
    size_t start_index = item_user.outerIndexPtr()[current_loc];
    size_t ep = item_user.outerIndexPtr()[current_loc + 1];
    size_t nnz = ep - start_index;
    if (nnz == 0) {
      return count;
    }
    for (size_t m = 0; m < n_count; m++) {
      for (size_t n = 0; n < n_step; n++) {
        current_loc = step_i_to_u(current_loc, rns);
        current_loc = step_u_to_i(current_loc, rns);
      }
      count[current_loc] += 1;
    }
    return count;
  };

  inline map<size_t, int> _run_item_walk_restart(float decay, size_t cutoff,
                                                 size_t item_start_index,
                                                 size_t n_count,
                                                 std::mt19937 &rns) const {
    map<size_t, int> count;
    auto current_loc = item_start_index;
    size_t start_index = item_user.outerIndexPtr()[current_loc];
    size_t ep = item_user.outerIndexPtr()[current_loc + 1];
    size_t nnz = ep - start_index;
    if (nnz == 0) {
      return count;
    }
    for (size_t m = 0; m < n_count; m++) {
      auto current_loc = item_start_index;
      for (size_t n = 0; n < cutoff; n++) {
        current_loc = step_i_to_u(current_loc, rns);
        current_loc = step_u_to_i(current_loc, rns);
        if (udist_(rns) < decay)
          break;
      }
      count[current_loc] += 1;
    }
    return count;
  };

private:
  CSRMatrix user_item, item_user;
  size_t n_item, n_user;
  // mt19937 random_state_;
};

} // namespace irspack

PYBIND11_MODULE(_rwr, m) {
  using namespace irspack;
  m.doc() = "Backend C++ inplementation for Random walk with restart.";
  py::class_<RandomWalkGenerator>(m, "RandomWalkGenerator")
      .def(py::init<CSRMatrix &>())
      .def("run_with_restart", &RandomWalkGenerator::run_with_restart);
}
