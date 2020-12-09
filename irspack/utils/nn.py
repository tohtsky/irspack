from dataclasses import dataclass
from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna import exceptions
from scipy import sparse as sps
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm


@dataclass
class MLPTrainingConfig:
    intermediate_dims: List[int]
    dropout: float = 0.0
    weight_decay: float = 0.0
    best_epoch: Optional[int] = None


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
            dropout = trial.suggest_float("dropout", 0, 1)

        if not self.tune_weight_decay:
            weight_decay = 0.0
        else:
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1)

        return MLPTrainingConfig(layer_dims, dropout, weight_decay)


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: MLPTrainingConfig):
        self.config = config
        super().__init__()
        dim_ins = [dim_in] + config.intermediate_dims
        dim_outs = config.intermediate_dims + [dim_out]
        layers: List[nn.Module] = []
        for din, dout in zip(dim_ins[:-1], dim_outs[:-1]):
            layers.extend(
                [nn.Dropout(p=config.dropout), nn.Linear(din, dout), nn.ReLU()]
            )
        layers.append(nn.Linear(dim_ins[-1], dim_outs[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        if sps.issparse(x):
            x_tensor = torch.tensor(x.astype(np.float32).toarray())
        else:
            x_tensor = torch.tensor(x.astype(np.float32))
        output = self.layers(x_tensor)
        return output


class MLPOptimizer(object):
    search_config: MLPSearchConfig
    best_trial_score: float
    best_config: Optional[MLPTrainingConfig]

    @staticmethod
    def stream(X: sps.csr_matrix, Y: sps.csc_matrix, mb_size: int, shuffle=True):
        assert X.shape[0] == Y.shape[0]
        shape_all: int = X.shape[0]
        index = np.arange(shape_all)
        if shuffle:
            np.random.shuffle(index)
        for start in range(0, shape_all, mb_size):
            end = min(shape_all, start + mb_size)
            mb_indices = index[start:end]
            yield X[mb_indices], Y[mb_indices], (end - start)

    def __init__(
        self,
        profile: sps.csr_matrix,
        embedding: np.ndarray,
        search_config: Optional[MLPSearchConfig] = None,
    ):

        profile_train, profile_test, embedding_train, embedding_test = train_test_split(
            profile.astype(np.float32), embedding.astype(np.float32), random_state=42
        )
        self.profile_train = profile_train
        self.profile_test = profile_test
        self.embedding_train = torch.tensor(embedding_train)
        self.embedding_test = torch.tensor(embedding_test)
        if search_config is None:
            self.search_config = MLPSearchConfig()
        else:
            self.search_config = search_config

    def search_best_config(
        self, n_trials: int = 10, logger: Optional[Logger] = None,
    ) -> Optional[MLPTrainingConfig]:
        self.best_trial_score = float("inf")
        self.best_config = None
        study = optuna.create_study()
        if logger is not None:
            r2 = (self.embedding_test.detach().numpy() ** 2).sum(axis=1).mean()
            logger.info(f"r2 baseline is {r2}")

        def objective(trial: optuna.Trial) -> float:
            config = self.search_config.suggest(trial)
            mlp = MLP(
                self.profile_train.shape[1], self.embedding_train.shape[1], config
            )
            score, epoch = self._ran_nn(mlp, trial=trial)
            config.best_epoch = epoch
            if score < self.best_trial_score:
                self.best_trial_score = score
                self.best_config = config
            return score

        study.optimize(objective, n_trials=n_trials)
        return self.best_config

    def search_param_fit_all(
        self, n_trials: int = 10, logger: Optional[Logger] = None,
    ) -> Tuple[MLP, MLPTrainingConfig]:
        best_param = self.search_best_config(n_trials, logger=logger)

        if best_param is None:
            raise RuntimeError("An error occurred during the optimization step.")

        mlp = self.fit_full(best_param)
        return mlp, best_param

    def _ran_nn(
        self, mlp: MLP, weight_decay: float = 0.0, trial: Optional[optuna.Trial] = None
    ) -> Tuple[float, int]:
        optimizer = torch.optim.Adam(mlp.parameters(), weight_decay=weight_decay)
        best_val_score = float("inf")
        best_model_filename = "best_model.pkl"
        n_epochs = 512
        train_index = np.arange(self.embedding_train.shape[0])
        mb_size = 128
        val_index = np.arange(self.embedding_test.shape[0])
        score_degradation_count = 0
        val_score_degradation_max = 10
        best_epoch = 0
        for epoch in tqdm(range(n_epochs)):
            mean_loss: List[float] = []
            mlp.train()
            for X_mb, y_mb, size in self.stream(
                self.profile_train, self.embedding_train, 128
            ):
                prediction = mlp(X_mb)
                optimizer.zero_grad()
                loss_train = (
                    ((prediction - torch.tensor(y_mb)) ** 2).sum(dim=1).mean(dim=0)
                )
                mean_loss.append(loss_train.detach().numpy())
                loss_train.backward()
                optimizer.step()

            mlp.eval()
            for X_mb, y_mb, size in self.stream(
                self.profile_test, self.embedding_test, 128, shuffle=False
            ):
                prediction = mlp(X_mb)
                optimizer.zero_grad()
                loss_val: torch.Tensor = ((prediction - torch.tensor(y_mb)) ** 2).sum(
                    dim=1
                ).mean(dim=0)
                mean_loss.append(loss_val.item())

            val_loss: float = np.mean(mean_loss)
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise exceptions.TrialPruned()

            if val_loss < best_val_score:
                best_epoch = epoch + 1
                best_val_score = val_loss
                torch.save(mlp.state_dict(), best_model_filename)
                score_degradation_count = 0
            else:
                score_degradation_count += 1

            if score_degradation_count >= val_score_degradation_max:
                break

        mlp.load_state_dict(torch.load("best_model.pkl"))
        mlp.eval()
        return best_val_score, best_epoch

    def fit_full(self, configs: MLPTrainingConfig):
        if configs.best_epoch is None:
            raise ValueError("best epoch not specified by MLP Config")
        mlp = MLP(self.profile_train.shape[1], self.embedding_train.shape[1], configs)

        optimizer = torch.optim.Adam(
            mlp.parameters(), weight_decay=configs.weight_decay
        )
        X = sps.vstack([self.profile_train, self.profile_test])
        y = torch.cat([self.embedding_train, self.embedding_test], 0)
        mb_size = 128
        for _ in tqdm(range(configs.best_epoch)):
            mlp.train()
            for X_mb, y_mb, size in self.stream(X, y, mb_size, True):
                prediction = mlp(X_mb)
                optimizer.zero_grad()
                loss = ((prediction - torch.tensor(y_mb)) ** 2).sum(dim=1).mean(dim=0)
                loss.backward()
                optimizer.step()
        return mlp
