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


class MovieLens20MDataManager(MovieLens100KDataManager):
    DOWNLOAD_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DEFAULT_PATH = os.path.expanduser("~/.ml-20m.zip")
    DATA_PATH_IN_ZIP = "ml-20m/ratings.csv"

    def _read_interaction(self, byte_stream: bytes):
        with BytesIO(byte_stream) as ifs:
            df = pd.read_csv(ifs)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
