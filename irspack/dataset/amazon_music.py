from .downloader import BaseDownloader
import os
from pathlib import Path
import urllib.request
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from gzip import GzipFile


class AmazonMusicDataManager(BaseDownloader):
    DEFAULT_PATH = Path.expanduser("~/.amazon-music.zip")

    def _save_to_zippath(self, path: os.PathLike) -> None:
        ratings_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv"
        meta_url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz"
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as zf:
            print("download rating...")
            with NamedTemporaryFile("w") as rating_temp:
                urllib.request.urlretrieve(ratings_url, rating_temp.name)
                rating_temp.seek(0)
                zf.write(rating_temp.name, "amazon-music/ratings.csv")

            print("download item meta data...")
            with NamedTemporaryFile("w") as meta_temp:
                urllib.request.urlretrieve(meta_url, meta_temp.name)
                meta_temp.seek(0)
                with GzipFile(meta_temp.name, "rb") as ifs:
                    zf.writestr("amazon-music/music-meta.json", ifs.read())

    def read_interaction(self) -> pd.DataFrame:
        return pd.read_csv(
            self._read_as_istream("amazon-music/ratings.csv"),
            header=None,
            names=["user_id", "music_id", "rating", "timestamp"],
        )

    def read_metadata(self) -> pd.DataFrame:
        results = []
        for l in self._read_as_istream("amazon-music/music-meta.json"):
            meta = eval(l.decode("utf-8"))
            results.append(
                (
                    meta["asin"],
                    meta.pop("title", None),
                    meta.pop("price", None),
                    meta.pop("categories", [None])[0],
                )
            )
        return pd.DataFrame(
            results, columns=["music_id", "title", "price", "categories"]
        ).set_index("music_id")
