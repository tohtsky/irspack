from abc import abstractmethod
from io import BytesIO

import pandas as pd

from irspack.dataset.downloader import SingleZipDownloader


class BaseMovieLenstDataLoader(SingleZipDownloader):
    def _read_as_istream(self, path: str) -> BytesIO:
        bytes_array = self.zf.read(path)
        return BytesIO(bytes_array)

    @abstractmethod
    def read_interaction(self) -> pd.DataFrame:
        raise NotImplementedError("Not implemented")  # pragma: no cover
