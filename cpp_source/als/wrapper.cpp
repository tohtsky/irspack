#include "IALSLearningConfig.hpp"
#include "IALSTrainer.hpp"
#include "pybind11/cast.h"
#include <Eigen/Sparse>
#include <cstddef>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace irspack::ials;
using std::vector;

PYBIND11_MODULE(_ials, m) {
  std::stringstream doc_stream;
  doc_stream << "irspack's core module for \"IALSRecommender\"." << std::endl
             << "Built to use" << std::endl
             << "\t" << Eigen::SimdInstructionSetsInUse();

  m.doc() = doc_stream.str();

  py::enum_<LossType>(m, "LossType")
      .value("Original", LossType::Original)
      .value("IALSPP", LossType::IALSPP)
      .export_values();

  auto model_config =
      py::class_<IALSModelConfig>(m, "IALSModelConfig")
          .def(py::init<size_t, Real, Real, Real, Real, int, LossType>())
          .def(py::pickle(
              [](const IALSModelConfig &config) {
                return py::make_tuple(config.K, config.alpha0, config.reg,
                                      config.nu, config.init_stdev,
                                      config.random_seed, config.loss_type);
              },
              [](py::tuple t) {
                if (t.size() != 7)
                  throw std::runtime_error("invalid state");

                size_t K = t[0].cast<size_t>();
                Real alpha0 = t[1].cast<Real>();
                Real reg = t[2].cast<Real>();
                Real nu = t[3].cast<Real>();
                Real init_stdev = t[4].cast<Real>();
                int random_seed = t[5].cast<int>();
                LossType loss_type = t[6].cast<LossType>();
                return IALSModelConfig(K, alpha0, reg, nu, init_stdev,
                                       random_seed, loss_type);
              }));
  py::class_<IALSModelConfig::Builder>(m, "IALSModelConfigBuilder")
      .def(py::init<>())
      .def("build", &IALSModelConfig::Builder::build)
      .def("set_K", &IALSModelConfig::Builder::set_K)
      .def("set_alpha0", &IALSModelConfig::Builder::set_alpha0)
      .def("set_reg", &IALSModelConfig::Builder::set_reg)
      .def("set_nu", &IALSModelConfig::Builder::set_nu)
      .def("set_init_stdev", &IALSModelConfig::Builder::set_init_stdev)
      .def("set_random_seed", &IALSModelConfig::Builder::set_random_seed)
      .def("set_loss_type", &IALSModelConfig::Builder::set_loss_type);

  auto solver_config =
      py::class_<SolverConfig>(m, "IALSSolverConfig")
          .def(py::init<size_t, bool, size_t>())
          .def(py::pickle(
              [](const SolverConfig &config) {
                return py::make_tuple(config.n_threads, config.use_cg,
                                      config.max_cg_steps);
              },
              [](py::tuple t) {
                if (t.size() != 3)
                  throw std::runtime_error("invalid state");

                size_t n_threads = t[0].cast<size_t>();
                bool use_cg = t[1].cast<bool>();
                size_t max_cg_steps = t[2].cast<size_t>();
                return SolverConfig(n_threads, use_cg, max_cg_steps);
              }));

  py::class_<SolverConfig::Builder>(m, "IALSSolverConfigBuilder")
      .def(py::init<>())
      .def("build", &SolverConfig::Builder::build)
      .def("set_n_threads", &SolverConfig::Builder::set_n_threads)
      .def("set_use_cg", &SolverConfig::Builder::set_use_cg)
      .def("set_max_cg_steps", &SolverConfig::Builder::set_max_cg_steps);

  py::class_<IALSTrainer>(m, "IALSTrainer")
      .def(py::init<IALSModelConfig, const SparseMatrix &>())
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
            IALSTrainer trainer(t[0].cast<IALSModelConfig>(),
                                t[1].cast<DenseMatrix>(),
                                t[2].cast<DenseMatrix>());
            return trainer;
          }));
}
