from dataclasses import dataclass
from functools import partial
from logging import Logger
from typing import Any, Callable, Iterator, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna
from jax._src.random import PRNGKey
from optuna import exceptions
from scipy import sparse as sps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from irspack.utils.default_logger import get_default_logger


@dataclass
class MLP:
    predict_function: Callable[[hk.Params, jnp.ndarray, bool], jnp.ndarray]
    params: hk.Params
    rng_key = PRNGKey(0)

    def predict(self, X: jnp.ndarray) -> np.ndarray:
        f: Callable[[hk.Params, PRNGKey, jnp.ndarray, bool], jnp.ndarray] = getattr(
            self, "predict_function"
        )
        return np.asarray(f(self.params, self.rng_key, X, False), dtype=np.float32)


@dataclass
class MLPTrainingConfig:
    intermediate_dims: List[int]
    dropout: float = 0.0
    weight_decay: float = 0.0
    best_epoch: Optional[int] = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
    learning_rate: float = 1e-3


@dataclass
class MLPSearchConfig:
    n_layers: int = 1
    tune_weight_decay: bool = False
    tune_dropout: bool = True
    layer_dim_max: int = 512

    def suggest(self, trial: optuna.Trial) -> MLPTrainingConfig:
        layer_dims = [
            trial.suggest_int(f"dim_{l}", 1, self.layer_dim_max)
            for l in range(self.n_layers)
        ]
        if not self.tune_dropout:
            dropout = 0.0
        else:
            dropout = trial.suggest_uniform("dropout", 0, 1)

        return MLPTrainingConfig(layer_dims, dropout)


def create_mlp(
    dim_out: int, config: MLPTrainingConfig
) -> Callable[[jnp.ndarray, bool], Any]:
    def mlp_function(X: jnp.ndarray, training: bool) -> Any:
        layers: List[Any] = []
        for d_o in config.intermediate_dims:
            if training:
                layers.append(
                    lambda x: hk.dropout(hk.next_rng_key(), config.dropout, x)
                )
            layers.append(hk.Linear(d_o))
            layers.append(config.activation)
        layers.append(hk.Linear(dim_out))
        return hk.Sequential(layers)(X)

    return mlp_function


class MLPOptimizer(object):
    search_config: MLPSearchConfig
    best_trial_score: float
    best_config: Optional[MLPTrainingConfig]

    @staticmethod
    def stream(
        X: sps.csr_matrix, Y: sps.csc_matrix, mb_size: int, shuffle: bool = True
    ) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, int]]:
        assert X.shape[0] == Y.shape[0]
        shape_all: int = X.shape[0]
        index = np.arange(shape_all)
        if shuffle:
            np.random.shuffle(index)
        for start in range(0, shape_all, mb_size):
            end = min(shape_all, start + mb_size)
            mb_indices = index[start:end]
            yield (
                jnp.asarray(X[mb_indices].toarray(), dtype=jnp.float32),
                jnp.asarray(Y[mb_indices], dtype=jnp.float32),
                (end - start),
            )

    def __init__(
        self,
        profile: sps.csr_matrix,
        embedding: np.ndarray,
        search_config: Optional[MLPSearchConfig] = None,
    ):

        (
            profile_train,
            profile_test,
            embedding_train,
            embedding_test,
        ) = train_test_split(
            profile.astype(np.float32),
            embedding.astype(np.float32),
            random_state=42,
        )
        self.profile_train = profile_train
        self.profile_test = profile_test
        self.embedding_train = jnp.asarray(embedding_train, dtype=jnp.float32)
        self.embedding_test = jnp.asarray(embedding_test, dtype=jnp.float32)
        if search_config is None:
            self.search_config = MLPSearchConfig()
        else:
            self.search_config = search_config

    def search_best_config(
        self,
        n_trials: int = 10,
        logger: Optional[Logger] = None,
        random_seed: Optional[int] = None,
    ) -> Optional[MLPTrainingConfig]:
        self.best_trial_score = float("inf")
        self.best_config = None
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=random_seed)
        )
        if logger is None:
            logger = get_default_logger()
        r2 = (self.embedding_test ** 2).mean(axis=1).mean()
        logger.info("MSE baseline is %f", r2)

        def objective(trial: optuna.Trial) -> float:
            config = self.search_config.suggest(trial)
            mlp_function = hk.transform(
                lambda x, training: (
                    create_mlp(
                        self.embedding_train.shape[1],
                        config,
                    )
                )(x, training)
            )
            score, epoch = self._train_nn_with_trial(
                mlp_function, config=config, trial=trial
            )
            config.best_epoch = epoch
            if score < self.best_trial_score:
                self.best_trial_score = score
                self.best_config = config
            return score

        study.optimize(objective, n_trials=n_trials)
        return self.best_config

    def search_param_fit_all(
        self,
        n_trials: int = 10,
        logger: Optional[Logger] = None,
        random_seed: Optional[int] = None,
    ) -> Tuple[MLP, MLPTrainingConfig]:
        best_param = self.search_best_config(
            n_trials, logger=logger, random_seed=random_seed
        )

        if best_param is None:
            raise RuntimeError("An error occurred during the optimization step.")

        mlp = self.fit_full(best_param)
        return mlp, best_param

    def _train_nn_with_trial(
        self,
        mlp: hk.Transformed,
        config: MLPTrainingConfig,
        trial: Optional[optuna.Trial] = None,
    ) -> Tuple[float, int]:

        rng_key = jax.random.PRNGKey(0)
        rng_key, sub_key = jax.random.split(rng_key)
        params = mlp.init(
            sub_key,
            jnp.zeros((1, self.profile_train.shape[1]), dtype=jnp.float32),
            False,
        )
        opt = optax.adam(config.learning_rate)
        opt_state = opt.init(params)

        rng_key, sub_key = jax.random.split(rng_key)

        @partial(jax.jit, static_argnums=(3,))
        def predict(
            params: hk.Params, rng: PRNGKey, X: jnp.ndarray, training: bool
        ) -> jnp.ndarray:
            return mlp.apply(params, rng, X, training)

        @partial(jax.jit, static_argnums=(4,))
        def loss_fn(
            params: hk.Params,
            rng: PRNGKey,
            X: jnp.ndarray,
            Y: jnp.ndarray,
            training: bool,
        ) -> jnp.ndarray:
            prediction = predict(params, rng, X, training)
            return ((Y - prediction) ** 2).mean(axis=1).sum()

        @jax.jit
        def update(
            params: hk.Params,
            rng: PRNGKey,
            opt_state: optax.OptState,
            X: jnp.ndarray,
            Y: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, hk.Params, optax.OptState]:
            loss_value = loss_fn(params, rng, X, Y, True)
            grad = jax.grad(loss_fn)(params, rng, X, Y, True)
            updates, opt_state = opt.update(grad, opt_state)
            new_params = optax.apply_updates(params, updates)
            return loss_value, new_params, opt_state

        best_val_score = float("inf")
        n_epochs = 512
        mb_size = 128
        score_degradation_count = 0
        val_score_degradation_max = 10
        best_epoch = 0
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0
            for X_mb, y_mb, _ in self.stream(
                self.profile_train, self.embedding_train, mb_size
            ):
                rng_key, sub_key = jax.random.split(rng_key)

                loss_value, params, opt_state = update(
                    params, sub_key, opt_state, X_mb, y_mb
                )
                train_loss += loss_value
            train_loss /= self.profile_train.shape[0]

            val_loss = 0
            for X_mb, y_mb, size in self.stream(
                self.profile_test, self.embedding_test, mb_size, shuffle=False
            ):
                val_loss += loss_fn(
                    params, rng_key, X_mb, y_mb, False
                )  # rng key will not be used
            val_loss /= self.profile_test.shape[0]
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise exceptions.TrialPruned()

            if val_loss < best_val_score:
                best_epoch = epoch + 1
                best_val_score = val_loss
                score_degradation_count = 0
            else:
                score_degradation_count += 1

            if score_degradation_count >= val_score_degradation_max:
                break

        return best_val_score, best_epoch

    def fit_full(self, config: MLPTrainingConfig) -> MLP:
        if config.best_epoch is None:
            raise ValueError("best epoch not specified by MLP Config")
        rng_key = jax.random.PRNGKey(0)

        mlp_function = hk.transform(
            lambda x, training: (
                create_mlp(
                    self.embedding_train.shape[1],
                    config,
                )
            )(x, training)
        )

        X = sps.vstack([self.profile_train, self.profile_test])
        y = jnp.concatenate([self.embedding_train, self.embedding_test], axis=0)
        mb_size = 128

        rng_key, sub_key = jax.random.split(rng_key)
        params = mlp_function.init(
            sub_key,
            jnp.zeros((1, self.profile_train.shape[1]), dtype=jnp.float32),
            True,
        )
        opt = optax.adam(config.learning_rate)
        opt_state = opt.init(params)

        @partial(jax.jit, static_argnums=(3,))
        def predict(
            params: hk.Params, rng: PRNGKey, X: jnp.ndarray, training: bool
        ) -> jnp.ndarray:
            return mlp_function.apply(params, rng, X, training)

        @partial(jax.jit, static_argnums=(4,))
        def loss_fn(
            params: hk.Params,
            rng: PRNGKey,
            X: jnp.ndarray,
            Y: jnp.ndarray,
            training: bool,
        ) -> jnp.ndarray:
            prediction = predict(params, rng, X, training)
            return ((Y - prediction) ** 2).mean(axis=1).sum()

        @jax.jit
        def update(
            params: hk.Params,
            rng: PRNGKey,
            opt_state: optax.OptState,
            X: jnp.ndarray,
            Y: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, hk.Params, optax.OptState]:
            loss_value = loss_fn(params, rng, X, Y, True)
            grad = jax.grad(loss_fn)(params, rng, X, Y, True)
            updates, opt_state = opt.update(grad, opt_state)
            new_params = optax.apply_updates(params, updates)
            return loss_value, new_params, opt_state

        mb_size = 128
        for _ in tqdm(range(config.best_epoch)):
            train_loss = 0
            for X_mb, y_mb, _ in self.stream(X, y, mb_size):
                rng_key, sub_key = jax.random.split(rng_key)
                loss_value, params, opt_state = update(
                    params, sub_key, opt_state, X_mb, y_mb
                )
                train_loss += loss_value
            train_loss /= self.profile_train.shape[0]
        return MLP(predict, params)
