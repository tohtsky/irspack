import os

import pandas as pd

from .base import BaseMovieLenstDataLoader


class MovieLens20MDataManager(BaseMovieLenstDataLoader):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-20m.zip")
    INTERACTION_PATH = "ml-20m/ratings.csv"

    def read_interaction(self) -> pd.DataFrame:
        with self._read_as_istream(self.INTERACTION_PATH) as ifs:
            df = pd.read_csv(ifs)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
