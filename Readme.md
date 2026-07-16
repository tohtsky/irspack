# irspack — implicit recommenders for practitioners

[![Python](https://img.shields.io/pypi/pyversions/irspack.svg)](https://pypi.org/project/irspack/)
[![PyPI](https://img.shields.io/pypi/v/irspack.svg)](https://pypi.org/project/irspack/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tohtsky/irspack/blob/main/LICENSE)
[![Build](https://github.com/tohtsky/irspack/workflows/Build/badge.svg?branch=main)](https://github.com/tohtsky/irspack/actions/workflows/wheels.yml)
[![Documentation](https://github.com/tohtsky/irspack/actions/workflows/docs.yml/badge.svg)](https://tohtsky.github.io/irspack/)
[![codecov](https://codecov.io/gh/tohtsky/irspack/branch/main/graph/badge.svg?token=kLgOKTQqcV)](https://codecov.io/gh/tohtsky/irspack)

**irspack** helps practitioners build, evaluate, and tune recommenders from
implicit feedback such as clicks, views, saves, and purchases.

It is designed for the work that happens before a recommender reaches
production: establishing strong baselines, comparing algorithms under the same
evaluation protocol, and tuning promising candidates without rewriting the
pipeline for every model.

- Practical recommenders including iALS, SLIM, item/user KNN, P3alpha,
  RP3beta, TopPop, and experimental feature-aware iALS
- Fast C++/Eigen implementations for core training and evaluation operations
- Consistent evaluation and Optuna-backed hyperparameter tuning
- Utilities for converting business IDs to sparse matrices and back
- Support for cold-start experiments with user and item side features

Read the [documentation](https://tohtsky.github.io/irspack/), see
[which recommender to try first](https://tohtsky.github.io/irspack/choosing_a_recommender.html),
or [start with your own interaction data](https://tohtsky.github.io/irspack/using_your_data.html).

## Installation

irspack requires Python 3.9 or later. Install the published package with:

```sh
pip install irspack
```

Pre-built wheels are published for supported Linux, macOS, and Windows
platforms. See [Installing from source](#installing-from-source) if a wheel is
not available for your environment or if you want CPU-specific compiler
optimizations.

## Quickstart

irspack consumes a SciPy sparse matrix whose rows are users and whose columns
are items. The values represent interaction strength; binary values are a good
default for events such as clicks or purchases.

```python
import numpy as np
import pandas as pd

from irspack import IALSRecommender, df_to_sparse

events = pd.DataFrame(
    {
        "user_id": ["alice", "alice", "bob", "bob", "carol", "carol"],
        "item_id": ["A", "B", "B", "C", "A", "D"],
    }
)

# Rows and columns in X correspond to user_ids and item_ids, respectively.
X, user_ids, item_ids = df_to_sparse(events, "user_id", "item_id")

model = IALSRecommender(X, n_components=8).learn()

# Recommend unseen items for Alice. Scores for already-seen items are -inf.
alice_index = np.flatnonzero(user_ids == "alice")[0]
scores = model.get_score_remove_seen(np.array([alice_index]))[0]
top_items = item_ids[np.argsort(scores)[::-1][:2]]
print(top_items)
```

For an offline comparison, split interactions into training and validation
matrices and evaluate every candidate with the same evaluator:

```python
from irspack import Evaluator, IALSRecommender, TopPopRecommender
from irspack.split import rowwise_train_test_split

X_train, X_validation = rowwise_train_test_split(
    X, test_ratio=0.2, ceil_n_heldout=True, random_state=0
)
evaluator = Evaluator(X_validation, cutoff=3)

for recommender_class in (TopPopRecommender, IALSRecommender):
    recommender = recommender_class(X_train).learn()
    print(recommender_class.__name__, evaluator.get_score(recommender)["ndcg"])
```

For timestamped events, prefer a temporal holdout over a random split. The
[using your own data guide](https://tohtsky.github.io/irspack/using_your_data.html)
shows the full workflow, including stable ID mappings and leakage-aware
evaluation.

## Which model should I try first?

| Situation | Good starting point |
| --- | --- |
| Sanity check and popularity baseline | `TopPopRecommender` |
| Strong general-purpose collaborative filtering | `IALSRecommender` |
| Explainable item-to-item recommendations | `CosineKNNRecommender` |
| Sparse implicit-feedback data | `RP3betaRecommender` or `SLIMRecommender` |
| Cold-start users/items with side information | Feature-aware iALS |

There is no universally best recommender. Start with a cheap baseline, compare
a small set of candidates using a split that reflects the product scenario,
then tune the winner. See the
[model selection guide](https://tohtsky.github.io/irspack/choosing_a_recommender.html)
for trade-offs and optional dependencies.

## Hyperparameter tuning

Every tunable recommender exposes the same Optuna-backed interface:

```python
best_params, trials = IALSRecommender.tune(
    X_train,
    evaluator,
    n_trials=20,
    random_seed=0,
)
best_model = IALSRecommender(X_train, **best_params).learn()
```

Iterative recommenders can use intermediate validation scores to stop
unpromising trials early.

## Optional recommenders

`BPRFMRecommender` wraps LightFM and requires a separate LightFM installation:

```sh
pip install lightfm
```

`MultVAERecommender` requires `jax`, `jaxlib`, `dm-haiku`, and `optax`. Follow
the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
if you need GPU support.

## Installing from source

A source build requires a C++17 compiler. To compile using the instruction set
available on the current machine:

```sh
CFLAGS="-march=native" pip install git+https://github.com/tohtsky/irspack.git
```

If installation fails, please
[open an issue](https://github.com/tohtsky/irspack/issues) and include your OS,
Python version, CPU architecture, and the complete build error.

## Development

This repository uses [uv](https://docs.astral.sh/uv/) for reproducible local
development:

```sh
uv sync
uv run pytest
```

Install documentation dependencies and build the site with:

```sh
uv sync --group docs
uv run sphinx-build -b html docs/source docs/_build/html
```

Use `uv lock --upgrade` only when intentionally updating dependencies. The lock
file is committed so local development and CI use the same versions.
