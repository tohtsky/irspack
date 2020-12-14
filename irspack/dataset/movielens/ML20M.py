import os
from io import BytesIO

import pandas as pd

from .ML100K import MovieLens100KDataManager


class MovieLens20MDataManager(MovieLens100KDataManager):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-20m.zip")
    DATA_PATH_IN_ZIP = "ml-20m/ratings.csv"

    def _read_interaction(self, byte_stream: bytes):
        with BytesIO(byte_stream) as ifs:
            df = pd.read_csv(ifs)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
