import os

import pandas as pd

from .base import BaseMovieLenstDataLoader


class MovieLens1MDataManager(BaseMovieLenstDataLoader):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-1m.zip")
    INTERACTION_PATH = "ml-1m/ratings.dat"
    ITEM_INFO_PATH = "ml-1m/movies.dat"
    USER_INFO_PATH = "ml-1m/users.dat"

    def read_interaction(self) -> pd.DataFrame:
        with self._read_as_istream(self.INTERACTION_PATH) as ifs:
            # This is a hack.
            # The true separator is "::", but this will force pandas
            # to use python engine, which is much slower.
            # instead we regard the separator to be ':' and imagine there is an empty (NaN).
            # values between "::"
            df = pd.read_csv(
                ifs,
                sep=":",
                header=None,
            )[[0, 2, 4, 6]].copy()

            df.columns = ["userId", "movieId", "rating", "timestamp"]
            df["timestamp"] = pd.to_datetime(df.timestamp, unit="s")
            return df

    def read_item_info(self) -> pd.DataFrame:
        with self._read_as_istream(self.ITEM_INFO_PATH) as ifs:
            data = pd.read_csv(
                ifs,
                sep="::",
                header=None,
                encoding="latin-1",
                names=["movieId", "title", "genres"],
                engine="python",
            )
            release_year = pd.to_numeric(
                data.title.str.extract(r"^.*\((?P<release_year>\d+)\)\s*$").release_year
            )
            data["release_year"] = release_year
            return data.set_index("movieId")

    def read_user_info(self) -> pd.DataFrame:
        with self._read_as_istream(self.USER_INFO_PATH) as ifs:
            return pd.read_csv(
                ifs,
                sep="::",
                header=None,
                names=["userId", "gender", "age", "occupation", "zipcode"],
                engine="python",
            ).set_index("userId")
