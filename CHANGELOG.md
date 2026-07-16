# Changelog

## 0.5.0 (2026-07-16)

### Added

- Added feature-aware iALS, including methods for learning and scoring user and
  item embeddings from side features.
- Added practical documentation for preparing interaction data, selecting a
  recommender, and mapping recommendation results back to business IDs.

### Changed

- Improved iALS and KNN training performance, including cache optimizations and
  Eigen 5 support.
- Moved the documentation site to GitHub Pages and refreshed its structure and
  styling.

### Breaking changes

- Removed `BaseRecommender.tune_with_study`. Pass an existing Optuna study to
  `BaseRecommender.tune` using the optional `study` argument instead.
- Removed the `fixed_params` argument from `BaseRecommender.tune`. Extra keyword
  arguments are now passed directly to the recommender constructor for every
  trial and override parameters suggested under the same names.
- Renamed the `random_seed` tuning argument to `tuning_random_seed`. The name
  `random_seed` can therefore be passed to recommender constructors without
  colliding with the Optuna sampler seed.
- Changed the best-parameters mapping returned by `BaseRecommender.tune` so it
  contains only suggested and learned parameters. Fixed recommender keyword
  arguments are no longer included and must be supplied separately when
  constructing the final recommender.
