import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from pytest_mock import MockFixture

from irspack.dataset import neu_mf

ZIPFILE_NAME = Path("~").expanduser() / "ml.test.zip"


class DummyNeuMFDownloader(neu_mf.NeuMFDownloader):
    TRAIN_URL: str = ""
    NEGATIVE_URL: str = ""


def test_newumf_dataset(mocker: MockFixture) -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")

    rng = np.random.default_rng(0)
    train_item_ids = {1: {1, 2, 3, 100}, 3: {2, 4, 101}, 4: {1, 4, 5, 102}}
    all_item_ids = sorted(list({z for y in train_item_ids.values() for z in y}))
    validation_true_positive = {1: 5, 3: 1, 4: 2}
    train_rows = []

    YEAR = 2021
    start_timestamp = datetime(YEAR, 1, 1).timestamp()

    for user_id, interacted_item_ids in train_item_ids.items():
        for item_id in interacted_item_ids:
            rating = (rng.integers(1, 5),)
            timestamp = int(start_timestamp) + 86400 * rng.integers(0, 364)
            train_rows.append(f"{user_id}\t{item_id}\t{rating}\t{timestamp}")
    train_bytes = "\n".join(train_rows).encode()
    test_bytes_list = []
    for user_id, interacted_item_ids in train_item_ids.items():
        validation_gt = validation_true_positive[user_id]
        untouched_item_set = (
            set(all_item_ids)
            .difference(interacted_item_ids)
            .difference({validation_gt})
        )
        samples = rng.choice(
            list(untouched_item_set),
            size=min(len(untouched_item_set), 3),
            replace=False,
        )
        test_bytes_list.append(
            f"({user_id},{validation_gt})\t" + "\t".join([f"{iid}" for iid in samples])
        )
    test_bytes = "\n".join(test_bytes_list).encode()

    mocker.patch(
        "irspack.dataset.neu_mf.urlopen",
        side_effect=[BytesIO(train_bytes), BytesIO(test_bytes)],
        autospec=True,
    )
    with TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "hoge.zip"
        dm = DummyNeuMFDownloader(save_path, force_download=True)
        train, test = dm.read_train_test()
    assert np.all(train["timestamp"].dt.year.values == YEAR)
    for row in train.itertuples():
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        assert item_id in train_item_ids[user_id]

    for row in test[test["positive"]].itertuples():
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        assert item_id == validation_true_positive[user_id]

    for row in test[~test["positive"]].itertuples():
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        assert item_id != validation_true_positive[user_id]
        assert item_id not in train_item_ids[user_id]
