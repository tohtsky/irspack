import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockFixture

from irspack.dataset import MINDDataManager


def _write_archive(path: Path, split: str, prefix: str = "") -> None:
    timestamp = "11/13/2019 8:36:57 AM" if split == "train" else "11/14/2019 9:00:00 AM"
    item_id = "N1" if split == "train" else "N3"
    entity = json.dumps([{"WikidataId": "Q1", "Label": "entity"}])
    with ZipFile(path, "w") as zf:
        zf.writestr(
            f"{prefix}behaviors.tsv",
            f"1\tU1\t{timestamp}\tN0\t{item_id}-1 N2-0\n",
        )
        zf.writestr(
            f"{prefix}news.tsv",
            f"{item_id}\tnews\tworld\tTitle\tAbstract\thttps://example.com\t{entity}\t[]\n",
        )
        zf.writestr(f"{prefix}entity_embedding.vec", "Q1\t0.1\t0.2\n")
        zf.writestr(f"{prefix}relation_embedding.vec", "R1\t0.3\t0.4\n")


@pytest.fixture
def manager(tmp_path: Path) -> MINDDataManager:
    train_path = tmp_path / "train.zip"
    dev_path = tmp_path / "dev.zip"
    _write_archive(train_path, "train")
    _write_archive(dev_path, "dev")
    result = MINDDataManager(train_path, dev_path)
    yield result
    result.close()


def test_read_behaviors(manager: MINDDataManager) -> None:
    behaviors = manager.read_behaviors("train")
    assert behaviors.loc[0, "user_id"] == "U1"
    assert behaviors.loc[0, "timestamp"] == pd.Timestamp("2019-11-13 08:36:57")


def test_read_interactions(manager: MINDDataManager) -> None:
    impressions = manager.read_impressions()
    assert impressions.shape == (4, 7)
    np.testing.assert_array_equal(impressions["clicked"], [True, False, True, False])
    np.testing.assert_array_equal(impressions["position"], [0, 1, 0, 1])

    interactions = manager.read_interaction()
    np.testing.assert_array_equal(interactions["item_id"], ["N1", "N3"])
    np.testing.assert_array_equal(interactions["split"], ["train", "dev"])


def test_read_item_info_and_embeddings(manager: MINDDataManager) -> None:
    items = manager.read_item_info()
    assert list(items.index) == ["N1", "N3"]
    assert items.loc["N1", "title_entities"][0]["WikidataId"] == "Q1"

    entities = manager.read_entity_embeddings()
    relations = manager.read_relation_embeddings()
    assert entities.dtypes.unique().tolist() == [np.dtype("float32")]
    np.testing.assert_allclose(entities.loc["Q1"], [0.1, 0.2])
    np.testing.assert_allclose(relations.loc["R1"], [0.3, 0.4])


def test_invalid_arguments(manager: MINDDataManager, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        manager.read_behaviors("test")
    with pytest.raises(ValueError, match="size"):
        MINDDataManager(tmp_path / "missing", tmp_path / "missing", size="tiny")
    with pytest.raises(ValueError, match="download_source"):
        MINDDataManager(
            tmp_path / "missing",
            tmp_path / "missing",
            download_source="invalid",
        )


def test_huggingface_download(mocker: MockFixture, tmp_path: Path) -> None:
    train_path = tmp_path / "train.zip"
    dev_path = tmp_path / "dev.zip"

    def fake_download(url: str, path: Path, hf_token: str) -> None:
        split = "train" if "train" in url else "dev"
        assert url.startswith(MINDDataManager.HUGGINGFACE_BASE_URL)
        assert hf_token == "secret"
        _write_archive(path, split, prefix=f"MINDsmall_{split}/")

    download = mocker.patch.object(
        MINDDataManager, "_download", side_effect=fake_download
    )
    manager = MINDDataManager(
        train_path,
        dev_path,
        force_download=True,
        hf_token="secret",
    )
    manager.close()
    assert download.call_count == 2
