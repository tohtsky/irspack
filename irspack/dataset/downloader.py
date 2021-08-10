import os
from typing import Optional
from abc import ABCMeta
from zipfile import ZipFile
import urllib.request
from io import BytesIO


class BaseZipDownloader(metaclass=ABCMeta):
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

