import os
import re
from typing import List, Tuple

import pandas as pd

from .base import BaseMovieLenstDataLoader


class MovieLens100KDataManager(BaseMovieLenstDataLoader):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-100k.zip")
    INTERACTION_PATH = "ml-100k/u.data"

    USER_INFO_PATH = "ml-100k/u.user"
    ITEM_INFO_PATH = "ml-100k/u.item"
    GENRE_PATH = "ml-100k/u.genre"

    def read_interaction(self) -> pd.DataFrame:
        with self._read_as_istream(self.INTERACTION_PATH) as ifs:
            data = pd.read_csv(
                ifs,
                sep="\t",
                header=None,
                names=["userId", "movieId", "rating", "timestamp"],
            )
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
            return data

    def read_user_info(self) -> pd.DataFrame:
        with self._read_as_istream(self.USER_INFO_PATH) as ifs:
            return pd.read_csv(
                ifs,
                sep="|",
                header=None,
                names=["userId", "age", "gender", "occupation", "zipcode"],
            ).set_index("userId")

    def _read_genre(self) -> List[str]:
        with self._read_as_istream(self.GENRE_PATH) as ifs:
            items = ifs.read().decode("latin-1").split()
            return [re.sub("\|\d+$", "", i.strip()) for i in items]

    def read_item_info(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with self._read_as_istream(self.ITEM_INFO_PATH) as ifs:
            df = pd.read_csv(ifs, sep="|", header=None, encoding="latin-1")

        genres = self._read_genre()
        df.columns = [
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "URL",
        ] + genres
        movie_ids = df.movieId.values
        df["release_date"] = pd.to_datetime(df.release_date)
        genre_df = pd.DataFrame(
            [
                dict(movieId=movie_ids[row], genre=genres[col])
                for row, col in zip(*df[genres].values.nonzero())
            ]
        )
        df = df.set_index("movieId")
        return df.drop(columns=genres), genre_df
