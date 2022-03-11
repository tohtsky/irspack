import urllib.request
from abc import ABCMeta, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile


class BaseDownloader(metaclass=ABCMeta):
    DEFAULT_PATH: Path
    _zf: Optional[ZipFile]

    @abstractmethod
    def _save_to_zippath(self, path: Path) -> None:
        raise NotImplementedError()

    def __init__(
        self, zippath: Optional[Union[Path, str]] = None, force_download: bool = False
    ):
        if zippath is None:
            zippath = self.DEFAULT_PATH
            if not zippath.exists():
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
                    self._save_to_zippath(zippath)
                    print("complete")
                else:
                    raise RuntimeError("could not read zipFile")

        self.zf = ZipFile(zippath)

    def _read_as_istream(self, path: str) -> BytesIO:
        bytes_array = self.zf.read(path)
        return BytesIO(bytes_array)


class SingleZipDownloader(BaseDownloader):
    DOWNLOAD_URL: str

    def _save_to_zippath(self, path: Path) -> None:
        urllib.request.urlretrieve(self.DOWNLOAD_URL, path)
