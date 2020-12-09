"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
from typing import IO, List, Optional

import numpy as np
from scipy import sparse as sps

from .base import BaseRecommenderWithColdStartPredictability
from ..definitions import (
    DenseScoreArray,
    InteractionMatrix,
    UserIndexArray,
)
from .base_earlystop import BaseRecommenderWithEarlyStopping, TrainerBase
from ..parameter_tuning import CategoricalSuggestion

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    warnings.warn(
        "failed to import torch. If you wish to use MultVAE, you have to install torch."
    )
    raise


class BaseMLP:
    ACTIVATOR_CLASS = torch.nn.Tanh
    classname = "BaseMLP"
    regularize = False

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: Optional[float] = None,
    ):

        super().__init__()
        input_dims = [input_dim] + hidden_dims
        output_dims = hidden_dims + [output_dim]
        self.lt_names: List[str] = []
        self.n_layers = len(input_dims)
        self.dropout: Optional[nn.Dropout]
        self.activator = self.ACTIVATOR_CLASS()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        for i, (d_in, d_out) in enumerate(zip(input_dims, output_dims)):
            name = f"{self.classname}_layer_{i}"

            setattr(self, name, nn.Linear(d_in, d_out))
            self.lt_names.append(name)

    def _lts(self) -> List[nn.Linear]:
        return [getattr(self, name) for name in self.lt_names]

    def weights_sq_sum(self) -> List[torch.Tensor]:
        return [(lt.weight ** 2).sum() for lt in self._lts()]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            h = self.dropout(h)
        for i, lt in enumerate(self._lts()):
            h = lt(h)
            if i < (self.n_layers - 1):
                h = self.activator(h)
        return h


class DecoderNN(BaseMLP, nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super().__init__(latent_dim, output_dim, hidden_dims)

    def forward(self, z):
        h = super().forward(z)
        return F.log_softmax(h, dim=1)


def l2_normalize(X):
    return X / torch.sqrt((X ** 2).sum(dim=1) + 1e-8).unsqueeze(1)


class EncoderNN(BaseMLP, nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout=0.5):
        self.latent_dim = latent_dim
        super().__init__(input_dim, 2 * latent_dim, hidden_dims, dropout=dropout)

    def forward(self, X):
        X = l2_normalize(X)
        h = super().forward(X)
        mu = h[:, : self.latent_dim]
        log_var = h[:, self.latent_dim :]
        return mu, log_var


class MultVAE(torch.nn.Module):
    def __init__(
        self,
        n_obs: int,
        latent_dim: int,
        enc_hidden_dims: List[int],
        dec_hidden_dims: List[int],
        dropout_p: float = 0.5,
        l2_reg: float = 0.01,
    ):
        super().__init__()
        self.q_network = EncoderNN(n_obs, latent_dim, enc_hidden_dims, dropout=0.5)
        self.p_network = DecoderNN(latent_dim, n_obs, dec_hidden_dims)
        self._kl_coeff = 0.0
        self.dropout_p = dropout_p
        self.l2_reg = l2_reg

    def set_kl_coeff(self, nv):
        self._kl_coeff = nv

    def kl_coeff(self):
        return self._kl_coeff

    def forward(self, X):

        mu, log_var = self.q_network(X)
        std = torch.exp(log_var * 0.5)

        # log_var = 2 * log (std)
        KL = (0.5 * (-log_var + std ** 2 + mu ** 2 - 1)).sum(dim=1).mean(dim=0)

        eps = torch.randn(mu.shape)
        z = mu + eps * std
        log_softmax = self.p_network(z)

        q_weights = sum(self.q_network.weights_sq_sum())
        p_weights = sum(self.p_network.weights_sq_sum())
        neg_ll = -(log_softmax * X).sum(dim=1).mean(dim=0)
        neg_elbo = neg_ll + self._kl_coeff * KL
        if self.l2_reg > 0:
            neg_elbo = neg_elbo + (q_weights + p_weights) * self.l2_reg
        return neg_elbo


class MultVAETrainer(TrainerBase):
    def __init__(
        self,
        X: InteractionMatrix,
        dim_z: int,
        enc_hidden_dims: List[int],
        dec_hidden_dims: Optional[List[int]],
        dropout_p: float,
        l2_regularizer: float,
        kl_anneal_goal: float,
        anneal_end_epoch: int,
        minibatch_size: int,
    ):
        self.X = X
        self.n_user = X.shape[0]
        self.n_item = X.shape[1]
        self.minibatch_size = minibatch_size
        self.kl_anneal_goal = kl_anneal_goal
        self.anneal_end_epoch = anneal_end_epoch
        if dec_hidden_dims is None:
            dec_hidden_dims = enc_hidden_dims
        self.vae = MultVAE(
            self.n_item,
            dim_z,
            enc_hidden_dims,
            dec_hidden_dims,
            dropout_p=dropout_p,
            l2_reg=l2_regularizer,
        )
        self.total_anneal_step = (anneal_end_epoch * self.n_user) / minibatch_size
        self.recommend_with_randomness = False

        self.optimizer = torch.optim.Adam(self.vae.parameters())
        self._update_count = 0

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        mb_arrays: List[DenseScoreArray] = []
        n_users: int = X.shape[0]
        self.vae.eval()
        X_csr: sps.csr_matrix = X.tocsr()
        with torch.no_grad():
            for mb_start in range(0, n_users, self.minibatch_size):
                mb_end = min(n_users, mb_start + self.minibatch_size)
                X_mb = X_csr[mb_start:mb_end].astype(np.float32).toarray()
                X_torch = torch.tensor(X_mb, requires_grad=False)
                mu, log_var = self.vae.q_network(X_torch)
                if self.recommend_with_randomness:
                    z = mu + torch.exp(log_var * 0.5) * torch.randn(mu.shape)
                else:
                    z = mu
                mb_arrays.append(self.vae.p_network(z).detach().numpy())
        score_concat = np.concatenate(mb_arrays, axis=0)
        return score_concat.astype(np.float64)

    def run_epoch(self, **kwargs) -> None:
        self.vae.train()
        user_indices = np.arange(self.n_user)
        np.random.shuffle(user_indices)
        mean_loss = []
        for mb_start in range(0, self.n_user, self.minibatch_size):
            current_kl_coeff = self.kl_anneal_goal * min(
                1, self._update_count / self.total_anneal_step
            )
            self.vae.set_kl_coeff(current_kl_coeff)

            mb_end = min(self.n_user, mb_start + self.minibatch_size)
            X = self.X[user_indices[mb_start:mb_end]].astype(np.float32)
            if sps.issparse(X):
                X = X.toarray()
            X_torch = torch.tensor(X, requires_grad=False)
            loss = self.vae(X_torch)
            self.optimizer.zero_grad()
            mean_loss.append(loss.detach().numpy())
            loss.backward()
            self.optimizer.step()
            self._update_count += 1

        loss = np.asarray(mean_loss).mean()

    def save_state(self, ofs: IO) -> None:
        torch.save(self.vae.state_dict(), ofs)

    def load_state(self, ifs: IO) -> None:
        self.vae.load_state_dict(torch.load(ifs))
        self.vae.eval()


class MultVAERecommender(
    BaseRecommenderWithEarlyStopping, BaseRecommenderWithColdStartPredictability
):
    default_tune_range = [
        CategoricalSuggestion("dim_z", [32, 64, 128, 256]),
        CategoricalSuggestion("enc_hidden_dims", [[128], [256], [512]]),
        CategoricalSuggestion("kl_anneal_goal", [0.1, 0.2, 0.4]),
    ]

    def __init__(
        self,
        X_all: InteractionMatrix,
        dim_z: int = 16,
        enc_hidden_dims: List[int] = [256],
        dec_hidden_dims: Optional[List[int]] = None,
        dropout_p: float = 0.5,
        l2_regularizer: float = 0,
        kl_anneal_goal: float = 0.2,
        anneal_end_epoch: int = 50,
        minibatch_size: int = 512,
        max_epoch: int = 300,
        validate_epoch: int = 5,
        score_degradation_max: int = 5,
        recommend_with_randomness=False,
    ) -> None:
        super().__init__(
            X_all,
            max_epoch=max_epoch,
            validate_epoch=validate_epoch,
            score_degration_max=score_degradation_max,
        )

        self.dim_z = dim_z
        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims
        self.kl_anneal_goal = kl_anneal_goal
        self.anneal_end_epoch = anneal_end_epoch
        self.minibatch_size = minibatch_size
        self.recommend_with_randomness = recommend_with_randomness
        self.dropout_p = dropout_p
        self.l2_regularizer = l2_regularizer

        self.trainer: Optional[MultVAETrainer] = None

    def create_trainer(self) -> MultVAETrainer:
        return MultVAETrainer(
            self.X_all,
            self.dim_z,
            self.enc_hidden_dims,
            self.dec_hidden_dims,
            self.dropout_p,
            self.l2_regularizer,
            self.kl_anneal_goal,
            self.anneal_end_epoch,
            self.minibatch_size,
        )

    def get_score(self, user_indices: UserIndexArray) -> DenseScoreArray:
        return self.get_score_cold_user(self.X_all[user_indices])

    def get_score_cold_user(self, X: InteractionMatrix) -> DenseScoreArray:
        if self.trainer is None:
            raise RuntimeError("encoder called before training.")
        return self.trainer.get_score_cold_user(X)
