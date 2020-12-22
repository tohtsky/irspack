from abc import ABC, abstractmethod, abstractmethod
from typing import Optional, Any
from zipfile import ZipFile
from io import BytesIO
import os
import urllib.request

import pandas as pd


class BaseMovieLenstDataLoader(ABC):
    DOWNLOAD_URL: str
    DEFAULT_PATH: str
    _zf: Optional[ZipFile]

    def __init__(self, zippath: Optional[str] = None):
        if zippath is None:
            zippath = self.DEFAULT_PATH
            if not os.path.exists(zippath):
                download = input(
                    "Could not find {}.\nCan I download and save it there?[y/N]".format(
                        zippath
                    )
                )
                if download.lower() == "y":
                    print("start download...")
                    urllib.request.urlretrieve(self.DOWNLOAD_URL, zippath)
                    print("complete")
                else:
                    raise RuntimeError("could not read zipFile")

        self.zf = ZipFile(zippath)

    def _read_as_istream(self, path: str) -> BytesIO:
        bytes_array = self.zf.read(path)
        return BytesIO(bytes_array)

    @abstractmethod
    def read_interaction(self) -> pd.DataFrame:
        raise NotImplementedError("Not implemented")
