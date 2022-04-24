import optuna

from irspack import Evaluator, IALSOptimizer, rowwise_train_test_split
from irspack.utils.sample_data import mf_example_data


def test_doubling_dimension_strategy() -> None:
    X = mf_example_data(100, 100, 8, random_state=0)
    storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    X_train, X_val = rowwise_train_test_split(X, random_state=0)
    optim = IALSOptimizer(X_train, Evaluator(X_val))
    SCALE = 1.1
    bp, df = optim.optimize_doubling_dimension(
        2,
        8,
        n_trials_initial=10,
        n_startup_trials_following=5,
        neighborhood_scale=SCALE,
        storage=storage,
    )
    tried_dimensions = sorted(df["n_components"].unique())
    assert len(tried_dimensions) == 3
    assert tried_dimensions[0] == 2
    assert tried_dimensions[1] == 4
    assert tried_dimensions[2] == 8

    study_name_2 = storage.get_study_name_from_id(1)
    assert study_name_2.endswith("_2")
    bp_2 = optuna.load_study(study_name_2, storage).best_params

    trials_dim_4 = df[df["n_components"] == 4]
    for dim_4_single_trials in trials_dim_4.itertuples():
        alpha0 = dim_4_single_trials.alpha0
        assert alpha0 <= (bp_2["alpha0"] * SCALE)
        assert alpha0 >= (bp_2["alpha0"] / SCALE)

        reg = dim_4_single_trials.reg
        assert reg <= (bp_2["reg"] * SCALE)
        assert reg >= (bp_2["reg"] / SCALE)

    study_name_4 = storage.get_study_name_from_id(2)
    assert study_name_4.endswith("_4")
    bp_4 = optuna.load_study(study_name_4, storage).best_params

    trials_dim_8 = df[df["n_components"] == 8]
    for dim_8_single_trials in trials_dim_8.itertuples():
        alpha0 = dim_8_single_trials.alpha0
        assert alpha0 <= (bp_4["alpha0"] * SCALE)
        assert alpha0 >= (bp_4["alpha0"] / SCALE)

        reg = dim_8_single_trials.reg
        assert reg <= (bp_4["reg"] * SCALE)
        assert reg >= (bp_4["reg"] / SCALE)

    assert bp["n_components"] == df.sort_values("value").iloc[0]["n_components"]
