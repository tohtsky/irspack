# irspack

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/irspack.svg)](https://pypi.python.org/pypi/irspack)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tohtsky/irspack)
[![Build](https://github.com/tohtsky/irspack/workflows/Build/badge.svg?branch=main)](https://github.com/tohtsky/irspack)
[![Read the Docs](https://readthedocs.org/projects/irspack/badge/?version=stable)](https://irspack.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/tohtsky/irspack/branch/main/graph/badge.svg?token=kLgOKTQqcV)](https://codecov.io/gh/tohtsky/irspack)

[**Docs**](https://irspack.readthedocs.io/en/latest/)

**irspack** is a Python package to train, evaluate, and optimize recommender systems based on implicit feedback.

There are already great packages for this purpose like

- [implicit](https://github.com/benfred/implicit)
- [daisyRec](https://github.com/AmazingDD/daisyRec)
- [RecSys2019_DeepLearning_Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) (which has influenced this project the most)

However, I decided to implement my own one to

- Use [optuna](https://github.com/optuna/optuna) for more efficient parameter search. In particular, if an early stopping scheme is available, optuna can prune unpromising trial based on the intermediate validation score, which drastically reduces overall running time for tuning.
- Use multi-threaded implementations of the number of algorithms (KNN and IALS) in C++.
- Deal with user cold-start scenarios using ["CB2CF" strategy](https://dl.acm.org/doi/10.1145/3298689.3347038), which I found very convenient in practice.

# Installation & Optional Dependencies

There are binaries for Linux, MacOS, and Windows with python>=3.6. You can install them via

```sh
pip install irspack
```

The binaries have been compiled to use AVX instruction. If you want to use AVX2/AVX512 or your environment does not support AVX, install it from source like

```sh
CFLAGS="-march=native" pip install git+https://github.com/tohtsky/irspack.git
```

In that case, you must have a decent version of C++ compiler (with C++11 support).

## Optional Dependencies

I have also prepared a wrapper class (`BPRFMRecommender`) to train and optimize BPR/warp loss Matrix factorization implemented in [lightfm](https://github.com/lyst/lightfm). To use it you have to install `lightfm` separately, e.g. by

```sh
pip install lightfm
```

If you want to use Mult-VAE and CB2CF features in cold-start scenarios, you'll need the following additional (pip-installable) packages:

- [jax](https://github.com/google/jax)
- [jaxlib](https://github.com/google/jax)
  - If you want to use GPU, follow the installation guide [https://github.com/google/jax#installation](https://github.com/google/jax#installation)
- [dm-haiku](https://github.com/deepmind/dm-haiku)
- [optax](https://github.com/deepmind/optax)

# Basic Usage

## Step 1. Train a recommender

We first represent the user/item interaction as a [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) matrix. Then we can feed it into our `Recommender` classes:

```Python
import numpy as np
import scipy.sparse as sps
from irspack.recommenders import P3alphaRecommender
from irspack.dataset.movielens import MovieLens100KDataManager

df = MovieLens100KDataManager().read_interaction()
unique_user_id, user_index = np.unique(df.userId, return_inverse=True)
unique_movie_id, movie_index = np.unique(df.movieId, return_inverse=True)
X_interaction = sps.csr_matrix(
  (np.ones(df.shape[0]), (user_index, movie_index))
)

recommender = P3alphaRecommender(X_interaction)
recommender.learn()

# for user 0 (whose userId is unique_user_id[0]),
# compute the masked score (i.e., already seen items have the score of negative infinity)
# of items.
recommender.get_score_remove_seen([0])
```

## Step 2. Evaluate on a validation set

We have to split the dataset to train and validation sets

```Python
from irspack.split import rowwise_train_test_split
from irspack.evaluator import Evaluator
X_train, X_val = rowwise_train_test_split(
    X_interaction, test_ratio=0.2, random_seed=0
)

# often, X_val is defined for a subset of users.
# `offset` specifies where the validated user blocks begin.
# In this split, X_val is defined for the same users as X_train,
# so offset = 0
evaluator = Evaluator(ground_truth=X_val, offset=0)

recommender = P3alphaRecommender(X_train)
recommender.learn()
evaluator.get_score(recommender)
```

This will print something like

```Python
{
  'appeared_item': 106.0,
  'entropy': 3.840445116672292,
  'gini_index': 0.9794929280523742,
  'hit': 0.8854718981972428,
  'map': 0.11283343078231302,
  'n_items': 1682.0,
  'ndcg': 0.3401244303579389,
  'precision': 0.27560975609756017,
  'recall': 0.19399215770339678,
  'total_user': 943.0,
  'valid_user': 943.0
}
```

## Step 3. Optimize the Hyperparameter

Now that we can evaluate the recommenders' performance against
the validation set, we can use [optuna](https://github.com/optuna/optuna)-backed hyperparameter optimizer.

```Python
from irspack.optimizers import P3alphaOptimizer

optimizer = P3alphaOptimizer(X_train, evaluator)
best_params, trial_dfs  = optimizer.optimize(n_trials=20)

# maximal ndcg around 0.38 ~ 0.39
trial_dfs.ndcg.max()
```

Of course, we have to hold-out another interaction set for test,
and measure the performance of tuned recommender against the test set.
See `examples/` for more complete examples.

# TODOs

- complete documentation
- more splitting schemes
- more benchmark dataset
