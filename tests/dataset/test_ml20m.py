from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np

from irspack.dataset.movielens import MovieLens20MDataManager


def test_ml20m() -> None:
    fp = NamedTemporaryFile("wb")
    fp.name
    with ZipFile(fp.name, "w") as zf:
        with zf.open("ml-20m/ratings.csv", "w") as ofs:
            ofs.write(
                """userId,movieId,rating,timestamp
1,2,5,0
1,3,5,86400
""".encode()
            )
    loader = MovieLens20MDataManager(fp.name)
    df = loader.read_interaction()
    np.testing.assert_array_equal(df["userId"].values, [1, 1])
    np.testing.assert_array_equal(df["movieId"].values, [2, 3])
    np.testing.assert_array_equal(df["rating"].values, [5, 5])
    np.testing.assert_array_equal(
        df["timestamp"].values,
        np.asarray(
            [
                "1970-01-01",
                "1970-01-02",
            ],
            dtype="datetime64[ns]",
        ),
    )
