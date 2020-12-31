import os
import urllib.request
from abc import ABCMeta, abstractmethod
from io import BytesIO
from typing import Any, Optional
from zipfile import ZipFile

import pandas as pd


class BaseMovieLenstDataLoader(object, metaclass=ABCMeta):
    DOWNLOAD_URL: str
    DEFAULT_PATH: str
    _zf: Optional[ZipFile]

    def __init__(self, zippath: Optional[str] = None, force_download: bool = False):
        if zippath is None:
            zippath = self.DEFAULT_PATH
            if not os.path.exists(zippath):
                if not force_download:
                    download = (
                        input(
                            "Could not find {}.\nCan I download and save it there?[y/N]".format(
                                zippath
                            )
                        ).lower()
                        == "y"
                    )
                else:
                    download = True
                if download:
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
        raise NotImplementedError("Not implemented")  # pragma: no cover
