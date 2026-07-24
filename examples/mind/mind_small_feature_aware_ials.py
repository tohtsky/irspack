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
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy import sparse as sps
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder

from irspack import (
    EvaluatorWithColdUser,
    IALSRecommender,
    TopPopRecommender,
    df_to_sparse,
)
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
    """Training data, warm-item features, and temporal evaluator."""

    X_train: sps.csr_matrix
    item_features_train: sps.csr_matrix
    evaluator: EvaluatorWithColdUser


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
    # Recommend only articles exposed during this evaluation period.
    candidate_item_ids = evaluation_impressions["item_id"].unique()

    # Keep warm items first so scoring can concatenate the model's native
    # scores with feature-derived cold-item scores.
    cold_item_ids = np.setdiff1d(candidate_item_ids, train_item_ids)
    eval_item_ids = np.concatenate([train_item_ids, cold_item_ids])
    n_warm_items = len(train_item_ids)
    metadata_item_ids = item_info.index.to_numpy()
    if np.setdiff1d(eval_item_ids, metadata_item_ids).size:
        raise RuntimeError("Some required items have no MIND article metadata.")

    if np.setdiff1d(target_events["item_id"].unique(), candidate_item_ids).size:
        raise RuntimeError("Some target items are outside the candidate catalog.")

    X_train, _, _ = df_to_sparse(
        pre_cutoff_events, "user_id", "item_id", item_ids=train_item_ids
    )
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

    recommendable_item_indices = pd.Index(eval_item_ids).get_indexer(candidate_item_ids)
    # This example evaluates item cold-start only.  Users without an interaction
    # on a trainable (warm) item cannot receive an iALS user embedding and are
    # intentionally excluded.
    has_history = np.asarray(X_history[:, :n_warm_items].getnnz(axis=1) > 0)
    X_history = X_history[has_history]
    X_target = X_target[has_history]
    if X_history.shape[0] == 0:
        raise RuntimeError("The evaluation period has no users with history.")

    evaluator_args = {
        "cutoff": EVALUATION_CUTOFF,
        "recommendable_items": recommendable_item_indices.tolist(),
        "mb_size": 256,
    }
    # Feature rows have the same warm-then-cold ordering as ``eval_item_ids``.
    item_features = make_item_features(item_info, train_item_ids, eval_item_ids)
    return TemporalEvaluationContext(
        X_train=X_train,
        item_features_train=item_features[:n_warm_items],
        evaluator=EvaluatorWithColdUser(
            X_history[:, :n_warm_items],
            X_target,
            cold_item_features=item_features[n_warm_items:],
            **evaluator_args,
        ),
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
        context.evaluator,
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
    include_top_pop: bool = False,
) -> Dict[str, Any]:
    """Train both iALS variants and evaluate all targets in the period."""
    models = {
        "iALS": train_model(context, baseline_config, False),
        "feature-aware iALS": train_model(context, feature_config, True),
    }
    scores_by_model = {
        model_name: context.evaluator.get_scores(model, [EVALUATION_CUTOFF])
        for model_name, model in models.items()
    }
    if include_top_pop:
        # TopPop is reported on test as an independent non-personalized
        # reference rather than participating in hyperparameter tuning.
        top_pop = TopPopRecommender(context.X_train).learn()
        scores_by_model["TopPop"] = context.evaluator.get_scores(
            top_pop, [EVALUATION_CUTOFF]
        )
    # Reorient model-first score dictionaries into metric-first JSON, which
    # makes side-by-side model comparisons straightforward.
    return {
        metric: {
            model_name: model_scores[metric]
            for model_name, model_scores in scores_by_model.items()
        }
        for metric in METRICS_TO_REPORT
    }


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
            include_top_pop=True,
        ),
    }
    # Write machine-readable output and echo the same payload for interactive
    # runs or CI logs.
    with args.output.open("w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
