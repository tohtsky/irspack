import re
from pathlib import Path
from typing import Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .downloader import BaseDownloader

_prefix = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


class NeuMFDownloader(BaseDownloader):
    TRAIN_URL: str
    NEGATIVE_URL: str
    _TRAIN_NAME = "train"
    _TEST_NAME = "test"

    def _save_to_zippath(self, path: Path) -> None:
        with ZipFile(path, "w") as save_zf:
            with save_zf.open(self._TRAIN_NAME, "w") as train_fs:
                b_train: bytes = urlopen(self.TRAIN_URL).read()
                train_fs.write(b_train)
            with save_zf.open(self._TEST_NAME, "w") as test_fs:
                b_test: bytes = urlopen(self.NEGATIVE_URL).read()
                test_fs.write(b_test)

    def read_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with self.zf.open(self._TRAIN_NAME) as train_fs:
            train_df: pd.DataFrame = pd.read_csv(
                train_fs,
                header=None,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
            )
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], unit="s")
        test_data = []
        with self.zf.open(self._TEST_NAME) as test_fs:
            for line_byte in test_fs:
                line = line_byte.decode()
                match_ = _prefix.search(line)
                assert match_ is not None
                uid_str, iid_str = match_.groups()
                uid = int(uid_str)
                iid = int(iid_str)
                test_data.append((uid, iid, True))
                for negative_iid in _prefix.sub("", line).strip().split("\t"):
                    test_data.append((uid, int(negative_iid), False))
        test_df = pd.DataFrame(
            test_data, columns=["user_id", "item_id", "positive"]
        ).drop_duplicates(["user_id", "item_id"], keep="first")
        return train_df, test_df


class NeuMFML1MDownloader(NeuMFDownloader):
    r"""Manages MovieLens 1M dataset split under 1-vs-100 negative evaluation protocol.

    Args:
        zippath:
            Where the zipped data is located. If `None`, assumes the path to be `~/.neumf-ml-1m.zip`.
            If the designated path does not exist, you will be prompted for the permission to download the data.
            Defaults to `None`.
        force_download:
            If `True`, the class will not prompt for the permission and start downloading immediately.
    """
    DEFAULT_PATH = Path("~/.neumf-ml-1m.zip").expanduser()

    TRAIN_URL = "https://raw.githubusercontent.com/tohtsky/neural_collaborative_filtering/master/Data/ml-1m.train.rating"
    NEGATIVE_URL = "https://raw.githubusercontent.com/tohtsky/neural_collaborative_filtering/master/Data/ml-1m.test.negative"


class NeuMFMPinterestDownloader(NeuMFDownloader):
    r"""Manages Pinterest dataset split under 1-vs-100 negative evaluation protocol.

    Args:
        zippath:
            Where the zipped data is located. If `None`, assumes the path to be `~/.neumf-pinterest.zip`.
            If the designated path does not exist, you will be prompted for the permission to download the data.
            Defaults to `None`.
        force_download:
            If `True`, the class will not prompt for the permission and start downloading immediately.
    """
    DEFAULT_PATH = Path("~/.neumf-pinterest.zip").expanduser()

    TRAIN_URL = "https://raw.githubusercontent.com/tohtsky/neural_collaborative_filtering/master/Data/pinterest-20.train.rating"
    NEGATIVE_URL = "https://raw.githubusercontent.com/tohtsky/neural_collaborative_filtering/master/Data/pinterest-20.test.negative"
