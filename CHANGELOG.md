# Changelog

## 0.5.1 (2026-07-24)

### Added

- Added `MINDDataManager` for downloading and reading the Microsoft News
  Dataset, including impressions, timestamped interactions, article metadata,
  and knowledge-graph embeddings.
- Added support for scoring and evaluating feature-only cold-start items with
  feature-aware iALS through
  `IALSRecommender.get_score_cold_user_with_item_features` and the
  `cold_item_features` argument of `EvaluatorWithColdUser`.
- Added `catalog_coverage` to evaluator results.
- Added streaming evaluation of consecutive score blocks through
  `Evaluator.get_score_from_score_chunks` and
  `Evaluator.get_scores_from_score_chunks`.
- Added an end-to-end MIND Small example for temporal evaluation of
  feature-aware iALS.

### Changed

- Parallelized feature-weight updates in feature-aware iALS using the
  configured number of threads.
- Expanded the feature-aware iALS documentation with cold-start evaluation
  guidance and the MIND Small experiment.

### Fixed

- Excluded unavailable (`-inf`) scores from recommendation candidates and
  handled evaluations with no recommendable items without invalid diversity
  calculations.

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
