# Implicit Feedback Recommender Systems

Notable features include:

- Use of [optuna](https://github.com/optuna/optuna) for more efficient parameter search. In particular, if an early stopping scheme is available, optuna can prune unpromising trial based on the intermediate validation score.
- multi-thread C++ implemantations are available for a number of algorithms
- Implement CB2CF strategy for cold-start scenarios.

# Installation

```
python setup.py install
```

If you want to use Mult-VAE, install pytorch by

```
pip install torch
```

# Usage

See `examples/`

## Todo
