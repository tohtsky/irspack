#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

namespace ials11 {
using Real = float;
using IndexType = std::size_t;
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
} // namespace ials11
