#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

namespace irspack {
namespace ials {
using Real = float;
using IndexType = std::size_t;
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using DenseMatrix =
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DenseVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
} // namespace ials
} // namespace irspack
