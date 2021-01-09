#include "IALSLearningConfig.hpp"
#include "IALSTrainer.hpp"
#include "pybind11/cast.h"
#include <Eigen/Sparse>
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace ials11;
using std::vector;

PYBIND11_MODULE(_ials, m) {
  std::stringstream doc_stream;
  doc_stream << "irspack's core module for \"IALSRecommender\"." << std::endl
             << "Built to use" << std::endl
             << "\t" << Eigen::SimdInstructionSetsInUse();

  m.doc() = doc_stream.str();
  py::class_<IALSLearningConfig>(m, "IALSLearningConfig")
      .def(py::init<size_t, Real, Real, Real, int, size_t, bool, size_t>())
      .def(py::pickle(
          [](const IALSLearningConfig &config) {
            return py::make_tuple(config.K, config.alpha, config.reg,
                                  config.init_stdev, config.n_threads,
                                  config.random_seed, config.use_cg,
                                  config.max_cg_steps);
          },
          [](py::tuple t) {
            if (t.size() != 8)
              throw std::runtime_error("invalid state");

            size_t K = t[0].cast<size_t>();
            Real alpha = t[1].cast<Real>();
            Real reg = t[2].cast<Real>();
            Real init_stdev = t[3].cast<Real>();
            size_t n_threads = t[4].cast<size_t>();
            int random_seed = t[5].cast<int>();
            bool use_cg = t[6].cast<bool>();
            size_t max_cg_steps = t[7].cast<size_t>();
            return IALSLearningConfig(K, alpha, reg, init_stdev, n_threads,
                                      random_seed, use_cg, max_cg_steps);
          }));
  py::class_<IALSLearningConfig::Builder>(m, "IALSLearningConfigBuilder")
      .def(py::init<>())
      .def("build", &IALSLearningConfig::Builder::build)
      .def("set_K", &IALSLearningConfig::Builder::set_K)
      .def("set_alpha", &IALSLearningConfig::Builder::set_alpha)
      .def("set_reg", &IALSLearningConfig::Builder::set_reg)
      .def("set_init_stdev", &IALSLearningConfig::Builder::set_init_stdev)
      .def("set_random_seed", &IALSLearningConfig::Builder::set_random_seed)
      .def("set_n_threads", &IALSLearningConfig::Builder::set_n_threads)
      .def("set_use_cg", &IALSLearningConfig::Builder::set_use_cg)
      .def("set_max_cg_steps", &IALSLearningConfig::Builder::set_max_cg_steps);

  py::class_<IALSTrainer>(m, "IALSTrainer")
      .def(py::init<IALSLearningConfig, const SparseMatrix &>())
      .def("step", &IALSTrainer::step)
      .def("user_scores", &IALSTrainer::user_scores)
      .def("transform_user", &IALSTrainer::transform_user)
      .def("transform_item", &IALSTrainer::transform_item)
      .def_readwrite("user", &IALSTrainer::user)
      .def_readwrite("item", &IALSTrainer::item)
      .def(py::pickle(
          [](const IALSTrainer &trainer) {
            return py::make_tuple(trainer.config_, trainer.user, trainer.item);
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state!");
            IALSTrainer trainer(t[0].cast<IALSLearningConfig>(),
                                t[1].cast<DenseMatrix>(),
                                t[2].cast<DenseMatrix>());
            return trainer;
          }));
}
