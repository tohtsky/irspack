from .downloader import BaseZipDownloader
import pandas as pd
import os


class CiteULikeADataManager(BaseZipDownloader):
    DOWNLOAD_URL = (
        "https://github.com/tohtsky/citeulike-a/archive/refs/heads/master.zip"
    )
    DEFAULT_PATH = os.path.expanduser("~/.citeulike-a.zip")

    def read_interaction(self) -> pd.DataFrame:
        ifs = self._read_as_istream("citeulike-a-master/users.dat")
        result = []
        cnt = 0
        for user_id, l in enumerate(ifs):
            cnt += 1
            user_cnt: int
            n_items = 0
            for i, id_ in enumerate(l.split()):
                if i == 0:
                    user_cnt = int(id_)
                else:
                    n_items += 1
                    result.append((user_id, int(id_)))
            assert user_cnt == n_items
        return pd.DataFrame(result, columns=["user_id", "item_id"])

    def read_item_meta(self) -> pd.DataFrame:
        item_meta = pd.read_csv(
            self._read_as_istream("citeulike-a-master/raw-data.csv"), encoding="latin-1"
        )
        item_meta.loc[:, "doc.id"] -= 1
        return item_meta.set_index("doc.id")
