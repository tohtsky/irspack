from abc import ABCMeta, abstractmethod
from io import BytesIO

import pandas as pd

from irspack.dataset.downloader import BaseZipDownloader


class BaseMovieLenstDataLoader(BaseZipDownloader, metaclass=ABCMeta):
    def _read_as_istream(self, path: str) -> BytesIO:
        bytes_array = self.zf.read(path)
        return BytesIO(bytes_array)

    @abstractmethod
    def read_interaction(self) -> pd.DataFrame:
        raise NotImplementedError("Not implemented")  # pragma: no cover
