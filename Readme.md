# irspack - Implicit recommender systems for practitioners

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/irspack.svg)](https://pypi.python.org/pypi/irspack)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tohtsky/irspack)
[![Build](https://github.com/tohtsky/irspack/workflows/Build/badge.svg?branch=main)](https://github.com/tohtsky/irspack)
[![Read the Docs](https://readthedocs.org/projects/irspack/badge/?version=stable)](https://irspack.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/tohtsky/irspack/branch/main/graph/badge.svg?token=kLgOKTQqcV)](https://codecov.io/gh/tohtsky/irspack)

[**Docs**](https://irspack.readthedocs.io/en/latest/)

**irspack** is a Python package for recommender systems based on implicit feedback, designed to be used by practitioners.

Some of its features include:

- Efficient parameter tuning enabled by C++/Eigen implementations of core recommender algorithms and [optuna](https://github.com/optuna/optuna).
  - In particular, if an early stopping scheme is available, optuna can prune out unpromising trial based on the intermediate validation scores.
- Various utility functions, including
  - ID/index mapping utilities
  - Fast, multithreaded argsort for batch recommendation retrieval
  - Efficient and configurable evaluation of recommender system performance

# Installation & Optional Dependencies

In most cases, you can install the pre-build binaries via

```sh
pip install irspack
```

The binaries have been compiled to use AVX instruction. If you want to use AVX2/AVX512 or your environment does not support AVX (like Rosetta 2 on Apple M1), install it from source like

```sh
CFLAGS="-march=native" pip install git+https://github.com/tohtsky/irspack.git
```

In that case, you must have a decent version of C++ compiler (with C++11 support). If it doesn't work, feel free to make an issue!

## Optional Dependencies

I have also prepared a wrapper class (`BPRFMRecommender`) to train/optimize BPR/warp loss Matrix factorization implemented in [lightfm](https://github.com/lyst/lightfm). To use it you have to install `lightfm` separately, e.g. by

```sh
pip install lightfm
```

If you want to use Mult-VAE, you'll need the following additional (pip-installable) packages:

- [jax](https://github.com/google/jax)
- [jaxlib](https://github.com/google/jax)
  - If you want to use GPU, follow the installation guide [https://github.com/google/jax#installation](https://github.com/google/jax#installation)
- [dm-haiku](https://github.com/deepmind/dm-haiku)
- [optax](https://github.com/deepmind/optax)

# Basic Usage

## Step 1. Train a recommender

To begin with, we represent the user/item interaction as a [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) matrix. Then we can feed it into recommender classes:

```Python
import numpy as np
import scipy.sparse as sps
from irspack import IALSRecommender, df_to_sparse
from irspack.dataset import MovieLens100KDataManager

df = MovieLens100KDataManager().read_interaction()

# Convert pandas.Dataframe into scipy's sparse matrix.
# The i'th row of `X_interaction` corresponds to `unique_user_id[i]`
# and j'th column of `X_interaction` corresponds to `unique_movie_id[j]`.
X_interaction, unique_user_id, unique_movie_id = df_to_sparse(
  df, 'userId', 'movieId'
)

recommender = IALSRecommender(X_interaction)
recommender.learn()

# for user 0 (whose userId is unique_user_id[0]),
# compute the masked score (i.e., already seen items have the score of negative infinity)
# of items.
recommender.get_score_remove_seen([0])
```

## Step 2. Evaluation on a validation set

To evaluate the performance of a recommenderm we have to split the dataset to train and validation sets:

```Python
from irspack.split import rowwise_train_test_split
from irspack.evaluation import Evaluator

# Random split
X_train, X_val = rowwise_train_test_split(
    X_interaction, test_ratio=0.2, random_state=0
)

evaluator = Evaluator(ground_truth=X_val)

recommender = IALSRecommender(X_train)
recommender.learn()
evaluator.get_score(recommender)
```

This will print something like

```Python
{
    'appeared_item': 435.0,
    'entropy': 5.160409123151053,
    'gini_index': 0.9198367595008214,
    'hit': 0.40084835630965004,
    'map': 0.013890322881619916,
    'n_items': 1682.0,
    'ndcg': 0.07867240014767263,
    'precision': 0.06797454931071051,
    'recall': 0.03327028758587522,
    'total_user': 943.0,
    'valid_user': 943.0
}
```

## Step 3. Hyperparameter optimization

Now that we can evaluate the recommenders' performance against the validation set, we can use [optuna](https://github.com/optuna/optuna)-backed hyperparameter optimization.

```Python
best_params, trial_dfs  = IALSRecommender.tune(X_train, evaluator, n_trials=20)

# maximal ndcg around 0.43 ~ 0.45
trial_dfs["ndcg@10"].max()
```

Of course, we have to hold-out another interaction set for test, and measure the performance of tuned recommender against the test set.

See `examples/` for more complete examples.

# TODOs

- more benchmark dataset
