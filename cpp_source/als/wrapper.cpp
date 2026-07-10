#include "IALSLearningConfig.hpp"
#include "IALSTrainer.hpp"
#include <Eigen/Sparse>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <tuple>

using namespace irspack::ials;
using namespace nanobind;

NB_MODULE(_ials_core, m) {
  std::stringstream doc_stream;
  doc_stream << "irspack's core module for \"IALSRecommender\"." << std::endl
             << "Built to use" << std::endl
             << "\t" << Eigen::SimdInstructionSetsInUse();

  m.doc() = doc_stream.str();

  nanobind::enum_<LossType>(m, "LossType")
      .value("ORIGINAL", LossType::ORIGINAL)
      .value("IALSPP", LossType::IALSPP);

  nanobind::enum_<SolverType>(m, "SolverType")
      .value("CHOLESKY", SolverType::Cholesky)
      .value("CG", SolverType::CG)
      .value("IALSPP", SolverType::IALSPP);

  // Preserve the legacy module-level enum values. IALSPP has historically
  // resolved to SolverType.IALSPP because SolverType was registered last.
  // Registering them after both enum classes keeps the generated stub valid.
  m.attr("ORIGINAL") = LossType::ORIGINAL;
  m.attr("CHOLESKY") = SolverType::Cholesky;
  m.attr("CG") = SolverType::CG;
  m.attr("IALSPP") = SolverType::IALSPP;

  auto model_config =
      nanobind::class_<IALSModelConfig>(m, "IALSModelConfig")
          .def(nanobind::init<size_t, Real, Real, Real, Real, int, LossType,
                              Real, Real, size_t>(),
               nanobind::arg("K"), nanobind::arg("alpha0"),
               nanobind::arg("reg"), nanobind::arg("nu"),
               nanobind::arg("init_stdev"), nanobind::arg("random_seed"),
               nanobind::arg("loss_type"),
               nanobind::arg("lambda_user_feature"),
               nanobind::arg("lambda_item_feature"),
               nanobind::arg("feature_warmup_epochs"))
          .def("__getstate__",
               [](const IALSModelConfig &config) {
                 return nanobind::make_tuple(
                     config.K, config.alpha0, config.reg, config.nu,
                     config.init_stdev, config.random_seed, config.loss_type,
                     config.lambda_user_feature,
                     config.lambda_item_feature,
                     config.feature_warmup_epochs);
               })
          .def("__setstate__",
               [](IALSModelConfig &ials_model_config,
                  const std::tuple<size_t, Real, Real, Real, Real, int,
                                   LossType, Real, Real, size_t> &state) {
                 new (&ials_model_config) IALSModelConfig(
                     std::get<0>(state), std::get<1>(state), std::get<2>(state),
                     std::get<3>(state), std::get<4>(state), std::get<5>(state),
                     std::get<6>(state), std::get<7>(state),
                     std::get<8>(state), std::get<9>(state));
               });
  nanobind::class_<IALSModelConfig::Builder>(m, "IALSModelConfigBuilder")
      .def(nanobind::init<>())
      .def("build", &IALSModelConfig::Builder::build)
      .def("set_K", &IALSModelConfig::Builder::set_K, nanobind::arg("K"))
      .def("set_alpha0", &IALSModelConfig::Builder::set_alpha0, nanobind::arg("alpha0"))
      .def("set_reg", &IALSModelConfig::Builder::set_reg, nanobind::arg("reg"))
      .def("set_nu", &IALSModelConfig::Builder::set_nu, nanobind::arg("nu"))
      .def("set_init_stdev", &IALSModelConfig::Builder::set_init_stdev, nanobind::arg("init_stdev"))
      .def("set_random_seed", &IALSModelConfig::Builder::set_random_seed, nanobind::arg("random_seed"))
      .def("set_loss_type", &IALSModelConfig::Builder::set_loss_type, nanobind::arg("loss_type"))
      .def("set_lambda_user_feature",
           &IALSModelConfig::Builder::set_lambda_user_feature,
           nanobind::arg("value"))
      .def("set_lambda_item_feature",
           &IALSModelConfig::Builder::set_lambda_item_feature,
           nanobind::arg("value"))
      .def("set_feature_warmup_epochs",
           &IALSModelConfig::Builder::set_feature_warmup_epochs,
           nanobind::arg("value"));

  auto solver_config =
      nanobind::class_<SolverConfig>(m, "IALSSolverConfig")
          .def(nanobind::init<size_t, SolverType, size_t, size_t, size_t>(),
               nanobind::arg("n_threads"), nanobind::arg("solver_type"),
               nanobind::arg("max_cg_steps"),
               nanobind::arg("ialspp_subspace_dimension"),
               nanobind::arg("ialspp_iteration"))
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
      .def("set_n_threads", &SolverConfig::Builder::set_n_threads, nanobind::arg("n_threads"))
      .def("set_solver_type", &SolverConfig::Builder::set_solver_type, nanobind::arg("solver_type"))
      .def("set_max_cg_steps", &SolverConfig::Builder::set_max_cg_steps, nanobind::arg("max_cg_steps"))
      .def("set_ialspp_subspace_dimension",
           &SolverConfig::Builder::set_ialspp_subspace_dimension,
           nanobind::arg("ialspp_subspace_dimension"))
      .def("set_ialspp_iteration",
           &SolverConfig::Builder::set_ialspp_iteration,
           nanobind::arg("ialspp_iteration"));

  nanobind::class_<IALSTrainer>(m, "IALSTrainer")
      .def(nanobind::init<IALSModelConfig, const SparseMatrix &>(),
           nanobind::arg("model_config"), nanobind::arg("interaction"))
      .def(nanobind::init<IALSModelConfig, const SparseMatrix &,
                          const FeatureMatrix &, const FeatureMatrix &>(),
           nanobind::arg("model_config"), nanobind::arg("interaction"),
           nanobind::arg("user_feature"), nanobind::arg("item_feature"))
      .def("step", &IALSTrainer::step, nanobind::arg("solver_config"))
      .def("user_scores", &IALSTrainer::user_scores, nanobind::arg("begin"),
           nanobind::arg("end"), nanobind::arg("solver_config"))
      .def("transform_user", &IALSTrainer::transform_user,
           nanobind::arg("interaction"), nanobind::arg("solver_config"))
      .def("transform_item", &IALSTrainer::transform_item,
           nanobind::arg("interaction"), nanobind::arg("solver_config"))
      .def("transform_user_with_feature",
           &IALSTrainer::transform_user_with_feature,
           nanobind::arg("interaction"), nanobind::arg("feature"),
           nanobind::arg("solver_config"))
      .def("transform_item_with_feature",
           &IALSTrainer::transform_item_with_feature,
           nanobind::arg("interaction"), nanobind::arg("feature"),
           nanobind::arg("solver_config"))
      .def("transform_user_feature", &IALSTrainer::transform_user_feature,
           nanobind::arg("feature"))
      .def("transform_item_feature", &IALSTrainer::transform_item_feature,
           nanobind::arg("feature"))
      .def("compute_loss", &IALSTrainer::compute_loss,
           nanobind::arg("solver_config"))
      .def_rw("user", &IALSTrainer::user)
      .def_rw("item", &IALSTrainer::item)
      .def_rw("user_feature_weight", &IALSTrainer::user_feature_weight)
      .def_rw("item_feature_weight", &IALSTrainer::item_feature_weight)
      .def("__getstate__", [](const IALSTrainer & trainer) {
            return nanobind::make_tuple(
                trainer.config_, trainer.user, trainer.item,
                trainer.user_feature_weight, trainer.item_feature_weight);
      })
      .def("__setstate__", [](IALSTrainer & trainer, nanobind::tuple state) {
          if (state.size() != 3 && state.size() != 5) {
            throw std::runtime_error("Invalid IALSTrainer pickle state.");
          }
          auto config = nanobind::cast<IALSModelConfig>(state[0]);
          auto user = nanobind::cast<DenseMatrix>(state[1]);
          auto item = nanobind::cast<DenseMatrix>(state[2]);
          new (&trainer) IALSTrainer(config, user, item);
          if (state.size() == 5) {
            trainer.user_feature_weight =
                nanobind::cast<DenseMatrix>(state[3]);
            trainer.item_feature_weight =
                nanobind::cast<DenseMatrix>(state[4]);
          }
      });
}
