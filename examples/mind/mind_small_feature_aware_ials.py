"""Compare ordinary and feature-aware iALS on a temporal MIND-small split.

The final day of the official train archive is used for validation, and the
official dev archive is used for testing.  Recommendation candidates are the
articles exposed during each evaluation period.  Only users with pre-cutoff
history are evaluated here.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy import sparse as sps
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder

from irspack import EvaluatorWithColdUser, IALSRecommender, df_to_sparse
from irspack.dataset import MINDDataManager

TRAIN_EPOCHS = 16
# Keep the ranking cutoff in one place because it is used both when constructing
# evaluators and when selecting the metric columns written to the final report.
EVALUATION_CUTOFF = 20
METRICS_TO_REPORT = (
    "ndcg@20",
    "recall@20",
    "hit@20",
    "gini_index@20",
    "entropy@20",
    "catalog_coverage@20",
)


@dataclass
class TemporalEvaluationContext:
    """Data, index mappings, and evaluators for one temporal evaluation split.

    ``X_train`` and the trained model use only warm items.  Evaluation matrices
    use ``eval_item_ids`` instead, which is the union of warm items and the
    evaluation period's candidate catalog.  The index arrays below bridge those
    two item spaces without adding zero-interaction columns to the training
    matrix.
    """

    X_train: sps.csr_matrix
    train_item_popularity: np.ndarray
    eval_item_ids: np.ndarray
    item_features_train: sps.csr_matrix
    item_features_eval: sps.csr_matrix
    warm_item_indices: np.ndarray
    cold_item_indices: np.ndarray
    recommendable_item_indices: np.ndarray
    evaluators: Dict[str, EvaluatorWithColdUser]


def make_item_features(
    item_info: pd.DataFrame,
    fit_item_ids: np.ndarray,
    output_item_ids: np.ndarray,
) -> sps.csr_matrix:
    """Fit feature transforms on warm items and apply them to requested items.

    ``fit_item_ids`` must contain only pre-cutoff items.  In particular, the
    transform is *not* refitted when ``output_item_ids`` includes cold items;
    doing so would leak statistics from the evaluation period.
    """
    # Reindexing, rather than filtering with ``isin``, preserves the exact row
    # order expected by all downstream item-index mappings.
    fit_info = item_info.reindex(fit_item_ids)
    output_info = item_info.reindex(output_item_ids)

    # Titles provide a compact content representation.  The vocabulary and IDF
    # weights are learned exclusively from warm items.
    vectorizer = TfidfVectorizer(
        max_features=30_000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )
    fit_text = vectorizer.fit_transform(fit_info["title"].fillna(""))
    # TruncatedSVD requires fewer components than both matrix dimensions.  The
    # dynamic cap also keeps this example valid for unusually small downloads.
    n_components = min(128, fit_text.shape[0] - 1, fit_text.shape[1] - 1)
    if n_components < 1:
        raise RuntimeError("Not enough pre-cutoff item text to fit item features.")

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    svd.fit(fit_text)
    # Apply the already-fitted text pipeline to every output item.  L2
    # normalization prevents title-vector magnitude from dominating the
    # categorical and bias features concatenated below.
    text_features = sps.csr_matrix(
        Normalizer(copy=False)
        .fit_transform(
            svd.transform(vectorizer.transform(output_info["title"].fillna("")))
        )
        .astype(np.float32)
    )
    categorical_columns = ["category", "subcategory"]
    # Unknown evaluation-period categories become all-zero categorical vectors
    # instead of causing a transform error.
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float32
    )
    encoder.fit(fit_info[categorical_columns].fillna("__missing__"))
    categorical = encoder.transform(
        output_info[categorical_columns].fillna("__missing__")
    )
    # The constant feature lets the learned feature map represent a global
    # latent-space offset even when other features are sparse or unseen.
    bias = sps.csr_matrix(np.ones((len(output_item_ids), 1), dtype=np.float32))
    return sps.hstack(
        [text_features, categorical, bias], format="csr", dtype=np.float32
    )


def build_evaluation_context(
    pre_cutoff_events: pd.DataFrame,
    target_events: pd.DataFrame,
    evaluation_impressions: pd.DataFrame,
    item_info: pd.DataFrame,
) -> TemporalEvaluationContext:
    """Build the complete leakage-safe context for one evaluation period."""
    # The same pre-cutoff clicks define both the collaborative training matrix
    # and each evaluated user's history.  Cold candidates must enter through
    # features, not as artificial all-zero columns in ``X_train``.
    train_item_ids = np.sort(pre_cutoff_events["item_id"].unique())
    train_item_popularity = (
        pre_cutoff_events["item_id"]
        .value_counts()
        .reindex(train_item_ids, fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    # Recommend only articles exposed during this evaluation period.
    candidate_item_ids = evaluation_impressions["item_id"].unique()

    # Evaluation needs warm items for history lookup plus all recommendable
    # candidates.  Both groups need metadata for feature construction.
    eval_item_ids = np.union1d(train_item_ids, candidate_item_ids)
    metadata_item_ids = item_info.index.to_numpy()
    if np.setdiff1d(eval_item_ids, metadata_item_ids).size:
        raise RuntimeError("Some required items have no MIND article metadata.")

    if np.setdiff1d(target_events["item_id"].unique(), candidate_item_ids).size:
        raise RuntimeError("Some target items are outside the candidate catalog.")

    X_train, _, _ = df_to_sparse(
        pre_cutoff_events, "user_id", "item_id", item_ids=train_item_ids
    )
    X_train = X_train.tocsr().astype(np.float32)
    # Multiple clicks express implicit preference, not increasing interaction
    # strength in this experiment, so collapse every observed pair to one.
    X_train.data[:] = 1

    # History and target matrices must share users and the expanded evaluation
    # item space.  Sorting users makes their row alignment deterministic.
    target_user_ids = np.sort(target_events["user_id"].unique())
    X_history, _, _ = df_to_sparse(
        pre_cutoff_events,
        "user_id",
        "item_id",
        user_ids=target_user_ids,
        item_ids=eval_item_ids,
    )
    X_target, _, _ = df_to_sparse(
        target_events,
        "user_id",
        "item_id",
        user_ids=target_user_ids,
        item_ids=eval_item_ids,
    )
    X_history = X_history.tocsr().astype(np.float32)
    X_target = X_target.tocsr().astype(np.float32)
    X_history.data[:] = 1
    X_target.data[:] = 1

    # Translate item IDs into columns of the expanded evaluation matrices.
    eval_item_index = pd.Series(np.arange(len(eval_item_ids)), index=eval_item_ids)
    warm_item_indices = eval_item_index.loc[train_item_ids].to_numpy(dtype=np.int64)
    recommendable_item_indices = eval_item_index.loc[candidate_item_ids].to_numpy(
        dtype=np.int64
    )
    # This example evaluates item cold-start only.  Users without an interaction
    # on a trainable (warm) item cannot receive an iALS user embedding and are
    # intentionally excluded.
    has_history = np.asarray(X_history[:, warm_item_indices].getnnz(axis=1) > 0)
    X_history = X_history[has_history]
    X_target = X_target[has_history]
    if X_history.shape[0] == 0:
        raise RuntimeError("The evaluation period has no users with history.")

    # Produce a target matrix containing only post-cutoff items.  It is used to
    # measure the capability that ordinary iALS and TopPop do not have.
    cold_target = X_target.copy().tolil()
    cold_target[:, warm_item_indices] = 0
    cold_target = cold_target.tocsr()
    has_cold_target = np.asarray(cold_target.getnnz(axis=1) > 0)
    if not has_cold_target.any():
        raise RuntimeError("No users with history have a new-item target.")

    # All target segments use the same full candidate catalog.  Segmenting
    # targets must not make the ranking task easier by shrinking its candidates.
    evaluator_args = {
        "cutoff": EVALUATION_CUTOFF,
        "recommendable_items": recommendable_item_indices.tolist(),
        "mb_size": 256,
    }
    cold_item_indices = np.setdiff1d(np.arange(len(eval_item_ids)), warm_item_indices)
    # Symmetrically isolate warm targets and restrict this evaluator to users
    # who actually have at least one warm positive in the period.
    warm_target = X_target.copy().tolil()
    warm_target[:, cold_item_indices] = 0
    warm_target = warm_target.tocsr()
    has_warm_target = np.asarray(warm_target.getnnz(axis=1) > 0)
    evaluators = {
        "all_targets": EvaluatorWithColdUser(X_history, X_target, **evaluator_args),
        "new_item_targets": EvaluatorWithColdUser(
            X_history[has_cold_target],
            cold_target[has_cold_target],
            **evaluator_args,
        ),
        "warm_item_targets": EvaluatorWithColdUser(
            X_history[has_warm_target],
            warm_target[has_warm_target],
            **evaluator_args,
        ),
    }
    # Fit one leakage-safe feature pipeline and slice warm rows for training.
    # Cold rows are retained only for feature-based inference at evaluation.
    item_features_eval = make_item_features(item_info, train_item_ids, eval_item_ids)
    return TemporalEvaluationContext(
        X_train=X_train,
        train_item_popularity=train_item_popularity,
        eval_item_ids=eval_item_ids,
        item_features_train=item_features_eval[warm_item_indices],
        item_features_eval=item_features_eval,
        warm_item_indices=warm_item_indices,
        cold_item_indices=cold_item_indices,
        recommendable_item_indices=recommendable_item_indices,
        evaluators=evaluators,
    )


def train_model(
    context: TemporalEvaluationContext, config: Dict[str, Any], feature_aware: bool
) -> IALSRecommender:
    """Train either model variant from an independently tuned configuration."""
    feature_args: Dict[str, Any] = {}
    if feature_aware:
        # Feature-aware iALS sees only warm-item feature rows during fitting.
        # Cold-item rows are transformed into embeddings later, during scoring.
        feature_args = {
            "item_features": context.item_features_train,
            "lambda_item_feature": config["lambda_item_feature"],
        }
    return IALSRecommender(
        context.X_train,
        n_components=config["n_components"],
        alpha0=config["alpha0"],
        reg=config["reg"],
        train_epochs=int(config.get("train_epochs", TRAIN_EPOCHS)),
        random_seed=0,
        **feature_args,
    ).learn()


def _metrics_from_scores(
    evaluator: EvaluatorWithColdUser,
    score_chunks: Iterable[np.ndarray],
    n_recommendable: int,
) -> Dict[str, float]:
    """Convert streamed score matrices into the metrics used by this example."""
    scores = evaluator.get_scores_from_score_chunks(score_chunks, [EVALUATION_CUTOFF])
    # ``appeared_item`` is a count; dividing by the eligible catalog size makes
    # coverage comparable across validation and test periods.
    scores["catalog_coverage@20"] = scores["appeared_item@20"] / n_recommendable
    return {name: scores[name] for name in METRICS_TO_REPORT}


def score_model(
    model: IALSRecommender,
    context: TemporalEvaluationContext,
    evaluator: EvaluatorWithColdUser,
    feature_aware: bool,
) -> Dict[str, float]:
    """Evaluate in batches without materializing the full MIND score matrix.

    The evaluator owns history in the expanded evaluation-item space, whereas
    the model was fitted in warm-item space.  The explicit index projections in
    this function keep those spaces aligned.
    """

    def chunks() -> Iterator[np.ndarray]:
        # A cold item has no collaborative factor.  Feature-aware iALS can infer
        # one from its metadata; ordinary iALS intentionally leaves it unscored.
        cold_embedding = None
        if feature_aware and context.cold_item_indices.size:
            cold_embedding = model.compute_item_embedding_from_features(
                context.item_features_eval[context.cold_item_indices]
            )
        for begin in range(0, evaluator.n_users, evaluator.mb_size):
            end = min(begin + evaluator.mb_size, evaluator.n_users)
            history = evaluator.input_interaction[begin:end]
            # Project each user's history onto the warm columns understood by
            # the fitted model before solving for a user embedding.
            user_embedding = model.compute_user_embedding(
                history[:, context.warm_item_indices]
            )
            # ``-inf`` makes every unsupported item impossible to recommend.
            # It also avoids accidentally treating a missing score as zero.
            scores = np.full(
                (end - begin, len(context.eval_item_ids)),
                -np.inf,
                dtype=user_embedding.dtype,
            )
            scores[:, context.warm_item_indices] = model.get_score_from_user_embedding(
                user_embedding
            )
            if cold_embedding is not None:
                # Warm-user × cold-item scores use the same latent-space inner
                # product as ordinary iALS once feature-derived factors exist.
                scores[:, context.cold_item_indices] = user_embedding.dot(
                    cold_embedding.T
                )
            yield scores

    return _metrics_from_scores(
        evaluator, chunks(), len(context.recommendable_item_indices)
    )


def score_top_pop(
    context: TemporalEvaluationContext, evaluator: EvaluatorWithColdUser
) -> Dict[str, float]:
    """Score every user with the same pre-cutoff item-popularity vector."""
    # Cold items have no training clicks, so TopPop cannot rank them.  Warm
    # popularity values are mapped into the expanded evaluation item space.
    popularity = np.full(len(context.eval_item_ids), -np.inf, dtype=np.float32)
    popularity[context.warm_item_indices] = context.train_item_popularity

    def chunks() -> Iterator[np.ndarray]:
        for begin in range(0, evaluator.n_users, evaluator.mb_size):
            end = min(begin + evaluator.mb_size, evaluator.n_users)
            # Repeating only one minibatch at a time preserves the same bounded
            # memory behavior as model-based scoring.
            yield np.repeat(popularity[None, :], end - begin, axis=0)

    return _metrics_from_scores(
        evaluator, chunks(), len(context.recommendable_item_indices)
    )


class TemporalTuningEvaluator(EvaluatorWithColdUser):
    """Connect temporal evaluation-context scoring to recommender tuning."""

    def __init__(self, context: TemporalEvaluationContext, feature_aware: bool):
        self.context = context
        self.feature_aware = feature_aware
        # ``IALSRecommender.tune`` reads these evaluator attributes directly.
        # Reuse their canonical definitions from the all-target evaluator.
        evaluator = context.evaluators["all_targets"]
        self.target_metric = evaluator.target_metric
        self.cutoff = evaluator.cutoff
        self.target_metric_name = evaluator.target_metric_name

    def get_score(self, model: IALSRecommender) -> Dict[str, float]:
        # Tuning follows the production-like catalog scoring path, including
        # feature-derived cold-item factors for the feature-aware variant.
        scores = score_model(
            model,
            self.context,
            self.context.evaluators["all_targets"],
            self.feature_aware,
        )
        # The tuning interface expects unqualified metric names because it
        # already knows the evaluator cutoff.
        return {name.split("@", 1)[0]: value for name, value in scores.items()}


def tune(
    context: TemporalEvaluationContext,
    feature_aware: bool,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Tune one model family and return its best parameters and trial history."""

    def suggest_parameters(trial: optuna.Trial) -> Dict[str, Any]:
        # Both model families search the same collaborative hyperparameters so
        # the comparison does not privilege one with a fixed configuration.
        parameters: Dict[str, Any] = {
            "n_components": trial.suggest_int("n_components", 32, 256, log=True),
            "alpha0": trial.suggest_float("alpha0", 0.03, 1.0, log=True),
            "reg": trial.suggest_float("reg", 1e-4, 1e-1, log=True),
        }
        if feature_aware:
            # The feature-map ridge penalty is meaningful only for the
            # feature-aware model and spans a deliberately broad log scale.
            parameters["lambda_item_feature"] = trial.suggest_float(
                "lambda_item_feature", 1.0, 1e6, log=True
            )
        return parameters

    fixed_parameters: Dict[str, Any] = {
        # Fix latent-factor initialization so trial-to-trial differences come
        # from hyperparameters rather than model randomness.
        "random_seed": 0,
    }
    if feature_aware:
        # Supplying features activates feature-aware training; the ordinary
        # baseline receives neither features nor its extra regularizer.
        fixed_parameters["item_features"] = context.item_features_train
    return IALSRecommender.tune(
        context.X_train,
        TemporalTuningEvaluator(context, feature_aware),
        n_trials=n_trials,
        tuning_random_seed=seed,
        parameter_suggest_function=suggest_parameters,
        validate_epoch=1,
        score_degradation_max=3,
        **fixed_parameters,
    )


def evaluate_period(
    context: TemporalEvaluationContext,
    baseline_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    segment_names: Tuple[str, ...] = ("all_targets", "new_item_targets"),
    include_top_pop: bool = False,
) -> Dict[str, Any]:
    """Train both iALS variants and evaluate each requested target segment."""
    # Models are trained once per evaluation context and reused across segments.
    # A segment changes only relevant targets/users, never fitting or candidates.
    models = {
        "iALS": (
            train_model(context, baseline_config, False),
            False,
        ),
        "feature-aware iALS": (
            train_model(context, feature_config, True),
            True,
        ),
    }
    segments: Dict[str, Any] = {}
    for segment_name in segment_names:
        evaluator = context.evaluators[segment_name]
        # Preserve whether a model supports feature-derived cold embeddings;
        # this flag controls only scoring, not target construction.
        scores_by_model = {
            model_name: score_model(model, context, evaluator, feature_aware)
            for model_name, (model, feature_aware) in models.items()
        }
        if include_top_pop:
            # TopPop is reported on test as an independent non-personalized
            # reference rather than participating in hyperparameter tuning.
            scores_by_model["TopPop"] = score_top_pop(context, evaluator)
        # Reorient model-first score dictionaries into metric-first JSON, which
        # makes side-by-side model comparisons straightforward.
        segments[segment_name] = {
            metric: {
                model_name: model_scores[metric]
                for model_name, model_scores in scores_by_model.items()
            }
            for metric in METRICS_TO_REPORT
        }
    return segments


def parse_args() -> argparse.Namespace:
    """Parse experiment controls while keeping data/split choices fixed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=Path("mind_feature_aware_ials_test.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # The context manager manages MIND archive download/extraction resources.
    # Interaction reads contain timestamped positive impressions only; the
    # untimestamped ``history`` field is intentionally not treated as events.
    with MINDDataManager(force_download=True) as manager:
        official_train = manager.read_interaction("train")
        official_dev = manager.read_interaction("dev")
        train_impressions = manager.read_impressions("train")
        dev_impressions = manager.read_impressions("dev")
        train_item_info = manager.read_item_info("train")
        all_item_info = manager.read_item_info()

    # Hold out the final calendar day of the official training archive.
    validation_start = official_train["timestamp"].max().floor("D")
    validation_train = official_train[official_train["timestamp"] < validation_start]
    validation_target = official_train[official_train["timestamp"] >= validation_start]
    validation_impressions = train_impressions[
        train_impressions["timestamp"] >= validation_start
    ]
    # Validation history and collaborative fitting both stop before the cutoff.
    # The candidate catalog comes from articles actually exposed on the held-out
    # day, including articles that had no earlier clicks.
    validation_context = build_evaluation_context(
        validation_train,
        validation_target,
        validation_impressions,
        train_item_info,
    )

    # Tune the two families independently.  Distinct Optuna seeds avoid making
    # their search trajectories artificially identical, while model training
    # itself remains deterministic via ``random_seed=0``.
    baseline_config, baseline_history = tune(
        validation_context, False, args.n_trials, seed=0
    )
    feature_config, feature_history = tune(
        validation_context, True, args.n_trials, seed=1
    )
    baseline_history.insert(0, "model", "iALS")
    feature_history.insert(0, "model", "feature-aware iALS")
    # Store every validation trial next to the final JSON so the selected
    # configuration and early-stopping behavior remain auditable.
    pd.concat([baseline_history, feature_history]).to_csv(
        args.output.with_name(f"{args.output.stem}_validation.csv"), index=False
    )

    # The official dev archive is chronologically later than official train, so
    # it serves as a single untouched test period.  Feature transforms are still
    # fitted on official-train items inside ``build_evaluation_context``.
    test_context = build_evaluation_context(
        official_train,
        official_dev,
        dev_impressions,
        all_item_info,
    )
    # Validation reports the all/new views used during model selection.  Test
    # additionally splits out warm targets and includes the TopPop reference.
    results = {
        "validation_start": validation_start.isoformat(),
        "model_config": {
            "iALS": baseline_config,
            "feature-aware iALS": feature_config,
        },
        "validation": evaluate_period(
            validation_context, baseline_config, feature_config
        ),
        "test": evaluate_period(
            test_context,
            baseline_config,
            feature_config,
            segment_names=(
                "all_targets",
                "warm_item_targets",
                "new_item_targets",
            ),
            include_top_pop=True,
        ),
    }
    # Write machine-readable output and echo the same payload for interactive
    # runs or CI logs.
    with args.output.open("w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
