import pickle
from dataclasses import dataclass
from typing import IO, Any, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import OptState, adam
from scipy import sparse as sps

from ..definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .base_earlystop import BaseRecommenderWithEarlyStopping, TrainerBase


class BaseMLP:
    def __init__(
        self,
        output_dim: int,
        hidden_dims: List[int],
    ):
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def __call__(
        self,
        X: jnp.ndarray,
    ) -> jnp.ndarray:
        layers = []
        for d_o in self.hidden_dims:
            layers.append(hk.Linear(d_o))
            layers.append(jnp.tanh)
        layers.append(hk.Linear(self.output_dim))

        return hk.Sequential(layers)(X)


class DecoderNN:
    def __init__(self, output_dim: int, hidden_dims: List[int]):
        mlp = BaseMLP(output_dim=output_dim, hidden_dims=hidden_dims)
        self.mlp = mlp

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.log_softmax(self.mlp(X), axis=1)


def l2_normalize(X: jnp.ndarray) -> jnp.ndarray:
    return X / jnp.sqrt((X ** 2).sum(axis=1) + 1e-8)[:, None]


class EncoderNN:
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
    ):
        self.mlp = BaseMLP(2 * latent_dim, hidden_dims)
        self.latent_dim = latent_dim

    def __call__(
        self, X: jnp.ndarray, dropout: float, train: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        X = l2_normalize(X)
        if train:
            X = hk.dropout(hk.next_rng_key(), dropout, X)
        h = self.mlp(X)
        mu = h[:, : self.latent_dim]
        log_var = h[:, self.latent_dim :]
        return mu, log_var


@dataclass
class MultVAEOutput:
    log_softmax: jnp.ndarray
    mean: jnp.ndarray
    log_stddev: jnp.ndarray
    KL: jnp.ndarray


class MultVAE:
    def __init__(
        self,
        n_obs: int,
        latent_dim: int,
        enc_hidden_dims: List[int],
        dec_hidden_dims: List[int],
        dropout_p: float = 0.5,
        l2_reg: float = 0.01,
    ):
        self.encoder_network = EncoderNN(
            latent_dim,
            enc_hidden_dims,
        )
        self.decoder_network = DecoderNN(n_obs, dec_hidden_dims)
        self._kl_coeff: float = 0.0
        self.dropout_p = dropout_p
        self.l2_reg = l2_reg

    def set_kl_coeff(self, nv: float) -> None:
        self._kl_coeff = nv

    def kl_coeff(self) -> float:
        return self._kl_coeff

    def __call__(self, X: jnp.ndarray, p: jnp.ndarray, train: bool) -> MultVAEOutput:
        mu, log_var = self.encoder_network(X, p, train)
        std = jnp.exp(log_var * 0.5)
        # log_var = 2 * log (std)
        KL: jnp.ndarray = 0.5 * (-log_var + std ** 2 + mu ** 2 - 1)
        KL = KL.sum(axis=1).mean(axis=0)

        if train:
            eps = jax.random.normal(hk.next_rng_key(), mu.shape)  # self.rng.rand
            z: jnp.ndarray = mu + eps * std
        else:
            z = mu
        log_softmax: jnp.ndarray = self.decoder_network(z)

        return MultVAEOutput(log_softmax, mu, log_var * 0.5, KL)


class MultVAETrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        dim_z: int,
        enc_hidden_dim: int,
        dec_hidden_dim: Optional[int],
        dropout_p: float,
        l2_regularizer: float,
        kl_anneal_goal: float,
        anneal_end_epoch: int,
        minibatch_size: int,
        learning_rate: float,
    ):
        self.X = X
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        self.minibatch_size = minibatch_size
        self.kl_anneal_goal = kl_anneal_goal
        self.anneal_end_epoch = anneal_end_epoch
        self.dropout_p = dropout_p
        self.l2_regularizer = l2_regularizer
        self.learning_rate = learning_rate

        self.rng_seq = hk.PRNGSequence(42)

        if dec_hidden_dim is None:
            dec_hidden_dim = enc_hidden_dim

        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.dim_z = dim_z

        self.total_anneal_step = (anneal_end_epoch * self.n_users) / minibatch_size
        self._setup_jax_funcs()

        self._update_count = 0
        params = self.vae_f.init(
            next(self.rng_seq),
            np.zeros((1, self.n_items), dtype=np.float32),
            self.dropout_p,
            True,
        )

        self.params = params

        self.opt_state = self.optimizer.init(self.params)

    def _setup_jax_funcs(self) -> None:
        vae_f = hk.transform(
            lambda X, p, train: MultVAE(
                self.n_items,
                self.dim_z,
                [self.enc_hidden_dim],
                [self.dec_hidden_dim],
                dropout_p=self.dropout_p,
                l2_reg=self.l2_regularizer,
            )(X, p, train)
        )
        self.optimizer = adam(learning_rate=self.learning_rate)

        def loss_fn(
            params: hk.Params,
            rng: jnp.ndarray,
            X: jnp.ndarray,
            kl_coeff: jnp.ndarray,
            dropout: float,
            train: bool,
        ) -> jnp.ndarray:
            mvresult: MultVAEOutput = vae_f.apply(params, rng, X, dropout, train)
            neg_ll = -(mvresult.log_softmax * X).sum(axis=1).mean()
            neg_elbo = neg_ll + kl_coeff * mvresult.KL
            return neg_elbo

        loss_fn = jax.jit(loss_fn, static_argnums=(4, 5))

        def update(
            params: hk.Params,
            rng: jnp.ndarray,
            opt_state: OptState,
            X: jnp.ndarray,
            kl_coeff: jnp.ndarray,
            dropout: float,
        ) -> Tuple[hk.Params, OptState]:
            grads = jax.grad(loss_fn)(params, rng, X, kl_coeff, dropout, True)
            updates, new_optstate = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(
                params,
                updates,
            )
            return new_params, new_optstate

        update = jax.jit(update, static_argnums=(5,))
        self.vae_f = vae_f
        self.update_function = update

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        mb_arrays: List[DenseScoreArray] = []
        n_users: int = X.shape[0]
        X_csr: sps.csr_matrix = X.tocsr()
        for mb_start in range(0, n_users, self.minibatch_size):
            mb_end = min(n_users, mb_start + self.minibatch_size)
            X_mb = X_csr[mb_start:mb_end].astype(np.float32).toarray()
            X_jax = jnp.asarray(X_mb)
            mvresult: MultVAEOutput = self.vae_f.apply(
                self.params, next(self.rng_seq), X_jax, self.dropout_p, False
            )

            mb_arrays.append(np.asarray(mvresult.log_softmax, dtype=np.float64))
        score_concat = np.concatenate(mb_arrays, axis=0)
        return score_concat

    def run_epoch(self) -> None:
        user_indices = np.arange(self.n_users)
        np.random.shuffle(user_indices)
        for mb_start in range(0, self.n_users, self.minibatch_size):
            current_kl_coeff = jnp.asarray(
                self.kl_anneal_goal
                * min(1, self._update_count / self.total_anneal_step)
            )
            mb_end = min(self.n_users, mb_start + self.minibatch_size)
            X = self.X[user_indices[mb_start:mb_end]].astype(np.float32)
            if sps.issparse(X):
                X = X.toarray()
            X_jax = jnp.asarray(X, dtype=jnp.float32)
            self.params, self.opt_state = self.update_function(
                self.params,
                next(self.rng_seq),
                self.opt_state,
                X_jax,
                current_kl_coeff,
                self.dropout_p,
            )
            self._update_count += 1

    def save_state(self, ofs: IO) -> None:
        pickle.dump(self.params, ofs)

    def load_state(self, ifs: IO) -> None:
        self.params = pickle.load(ifs)

    def __getstate__(self) -> Any:
        serealized = dict(**self.__dict__)
        serealized.pop("update_function", None)
        serealized.pop("vae_f", None)
        serealized.pop("optimizer")
        return serealized

    def __setstate__(self, state: Dict[str, Any]) -> Any:
        self.__dict__.update(state)
        self._setup_jax_funcs()


class MultVAERecommender(BaseRecommenderWithEarlyStopping):
    def __init__(
        self,
        X_train_all: InteractionMatrix,
        dim_z: int = 16,
        enc_hidden_dims: int = 256,
        dec_hidden_dims: Optional[int] = None,
        dropout_p: float = 0.5,
        l2_regularizer: float = 0,
        kl_anneal_goal: float = 0.2,
        anneal_end_epoch: int = 50,
        minibatch_size: int = 512,
        max_epoch: int = 300,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            X_train_all,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degradation_max=score_degradation_max,
        )

        self.dim_z = dim_z
        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims
        self.kl_anneal_goal = kl_anneal_goal
        self.anneal_end_epoch = anneal_end_epoch
        self.minibatch_size = minibatch_size
        self.dropout_p = dropout_p
        self.l2_regularizer = l2_regularizer
        self.learning_rate = learning_rate

        self.trainer: Optional[MultVAETrainer] = None

    def _create_trainer(self) -> MultVAETrainer:
        return MultVAETrainer(
            self.X_train_all,
            self.dim_z,
            self.enc_hidden_dims,
            self.dec_hidden_dims,
            self.dropout_p,
            self.l2_regularizer,
            self.kl_anneal_goal,
            self.anneal_end_epoch,
            self.minibatch_size,
            self.learning_rate,
        )

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.get_score_cold_user(self.X_train_all[user_indices])

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("encoder called before training.")
        return self.trainer.get_score_cold_user(X)
