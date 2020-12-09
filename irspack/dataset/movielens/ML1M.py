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
import os
from io import BytesIO

import pandas as pd

from .ML100K import MovieLens100KDataManager


class MovieLens1MDataManager(MovieLens100KDataManager):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-1m.zip")
    DATA_PATH_IN_ZIP = "ml-1m/ratings.dat"
    ITEM_INFO_PATH = "ml-1m/movies.dat"
    USER_INFO_PATH = "ml-1m/users.dat"

    def _read_interaction(self, byte_stream: bytes) -> pd.DataFrame:
        with BytesIO(byte_stream) as ifs:
            df = pd.read_csv(
                ifs,
                sep="\:\:",
                header=None,
                names=["userId", "movieId", "rating", "timestamp"],
                engine="python",
            )
            df["timestamp"] = pd.to_datetime(df.timestamp, unit="s")
            return df

    def _read_item_info(self, byte_stream: bytes) -> pd.DataFrame:
        assert self.zf is not None
        with BytesIO(byte_stream) as ifs:
            data = pd.read_csv(
                ifs,
                sep="::",
                header=None,
                encoding="latin-1",
                names=["movieId", "title", "genres"],
            )
            release_year = pd.to_numeric(
                data.title.str.extract(r"^.*\((?P<release_year>\d+)\)\s*$").release_year
            )
            data["release_year"] = release_year
            return data.set_index("movieId")

    def _read_user_info(self, byte_stream: bytes) -> pd.DataFrame:
        with BytesIO(byte_stream) as ifs:
            return pd.read_csv(
                ifs,
                sep="::",
                header=None,
                names=["userId", "gender", "age", "occupation", "zipcode"],
            ).set_index("userId")
