from abc import abstractmethod
from io import BytesIO

import pandas as pd

from ..downloader import SingleZipDownloader


class BaseMovieLenstDataLoader(SingleZipDownloader):
    def _read_as_istream(self, path: str) -> BytesIO:
        bytes_array = self.zf.read(path)
        return BytesIO(bytes_array)

    @abstractmethod
    def read_interaction(self) -> pd.DataFrame:
        r"""Reads the entire user/movie/rating/timestamp interaction data.

        Returns:
            The interaction `pd.DataFrame`, whose columns are
            `["userId", "movieId", "rating", "timestamp"]`.
        """
        raise NotImplementedError("Not implemented")  # pragma: no cover
