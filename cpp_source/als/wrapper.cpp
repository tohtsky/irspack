#include "IALSLearningConfig.hpp"
#include "IALSTrainer.hpp"
#include <Eigen/Sparse>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <tuple>

using namespace irspack::ials;
using namespace nanobind;

NB_MODULE(_ials_core, m) {
  std::stringstream doc_stream;
  doc_stream << "irspack's core module for \"IALSRecommender\"." << std::endl
             << "Built to use" << std::endl
             << "\t" << Eigen::SimdInstructionSetsInUse();

  // m.doc() = doc_stream.str();

  nanobind::enum_<LossType>(m, "LossType")
      .value("ORIGINAL", LossType::ORIGINAL)
      .value("IALSPP", LossType::IALSPP)
      .export_values();

  nanobind::enum_<SolverType>(m, "SolverType")
      .value("CHOLESKY", SolverType::Cholesky)
      .value("CG", SolverType::CG)
      .value("IALSPP", SolverType::IALSPP)
      .export_values();

  auto model_config =
      nanobind::class_<IALSModelConfig>(m, "IALSModelConfig")
          .def(nanobind::init<size_t, Real, Real, Real, Real, int, LossType>())
          .def("__getstate__",
               [](const IALSModelConfig &config) {
                 return nanobind::make_tuple(
                     config.K, config.alpha0, config.reg, config.nu,
                     config.init_stdev, config.random_seed, config.loss_type);
               })
          .def("__setstate__",
               [](IALSModelConfig &ials_model_config,
                  const std::tuple<size_t, Real, Real, Real, Real, int,
                                   LossType> &state) {
                 new (&ials_model_config) IALSModelConfig(
                     std::get<0>(state), std::get<1>(state), std::get<2>(state),
                     std::get<3>(state), std::get<4>(state), std::get<5>(state),
                     std::get<6>(state));
               });
  nanobind::class_<IALSModelConfig::Builder>(m, "IALSModelConfigBuilder")
      .def(nanobind::init<>())
      .def("build", &IALSModelConfig::Builder::build)
      .def("set_K", &IALSModelConfig::Builder::set_K)
      .def("set_alpha0", &IALSModelConfig::Builder::set_alpha0)
      .def("set_reg", &IALSModelConfig::Builder::set_reg)
      .def("set_nu", &IALSModelConfig::Builder::set_nu)
      .def("set_init_stdev", &IALSModelConfig::Builder::set_init_stdev)
      .def("set_random_seed", &IALSModelConfig::Builder::set_random_seed)
      .def("set_loss_type", &IALSModelConfig::Builder::set_loss_type);

  auto solver_config =
      nanobind::class_<SolverConfig>(m, "IALSSolverConfig")
          .def(nanobind::init<size_t, SolverType, size_t, size_t, size_t>())
          .def(
            "__getstate__", [](const SolverConfig &config) {
                return std::make_tuple(
                    config.n_threads, config.solver_type, config.max_cg_steps,
                    config.ialspp_subspace_dimension, config.ialspp_iteration);
              }
          )
          .def(
            "__setstate__", [](SolverConfig &config, const std::tuple<size_t, SolverType, size_t, size_t, size_t> &state) {
              new (&config) SolverConfig(
                std::get<0>(state),
                std::get<1>(state),
                std::get<2>(state),
                std::get<3>(state),
                std::get<4>(state));
            }
          );

  nanobind::class_<SolverConfig::Builder>(m, "IALSSolverConfigBuilder")
      .def(nanobind::init<>())
      .def("build", &SolverConfig::Builder::build)
      .def("set_n_threads", &SolverConfig::Builder::set_n_threads)
      .def("set_solver_type", &SolverConfig::Builder::set_solver_type)
      .def("set_max_cg_steps", &SolverConfig::Builder::set_max_cg_steps)
      .def("set_ialspp_subspace_dimension",
           &SolverConfig::Builder::set_ialspp_subspace_dimension)
      .def("set_ialspp_iteration",
           &SolverConfig::Builder::set_ialspp_iteration);

  nanobind::class_<IALSTrainer>(m, "IALSTrainer")
      .def(nanobind::init<IALSModelConfig, const SparseMatrix &>())
      .def("step", &IALSTrainer::step)
      .def("user_scores", &IALSTrainer::user_scores)
      .def("transform_user", &IALSTrainer::transform_user)
      .def("transform_item", &IALSTrainer::transform_item)
      .def("compute_loss", &IALSTrainer::compute_loss)
      .def_rw("user", &IALSTrainer::user)
      .def_rw("item", &IALSTrainer::item)
      .def("__getstate__", [](const IALSTrainer & trainer) {
            return std::make_tuple(trainer.config_, trainer.user, trainer.item);
      })
      .def("__setstate__", [](IALSTrainer & trainer, const std::tuple<IALSModelConfig, DenseMatrix, DenseMatrix> & state) {
          new (&trainer) IALSTrainer(
            std::get<0>(state),             std::get<1>(state),
            std::get<2>(state)

          );
      });
}
