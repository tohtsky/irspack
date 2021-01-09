import os
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from irspack.dataset.movielens import (
    MovieLens1MDataManager,
    MovieLens20MDataManager,
    MovieLens100KDataManager,
)

ZIPFILE_NAME = os.path.join(os.path.expanduser("~"), "ml.test.zip")


def test_ml100k() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")
    GENRES = ["fantasy", "action", "thriller"]
    try:
        with ZipFile(ZIPFILE_NAME, "w") as zf:
            with zf.open("ml-100k/u.data", "w") as ofs:
                ofs.write("1\t2\t5\t0\n1\t3\t5\t86400".encode())

            with zf.open("ml-100k/u.genre", "w") as ofs:
                genre_string = ""
                for i, genre in enumerate(GENRES):
                    genre_string += f"{genre}|{i}\n"
                ofs.write(genre_string.encode())

            with zf.open("ml-100k/u.item", "w") as ofs:
                # movieId = 1 has action tag,
                # movieId = 2 has fantasy & thriller tags
                ofs.write(
                    """1|A fantastic movie|2020-01-01|2021-01-01|http://example.com|0|1|0
    2|Pandemic|2020-01-01|2021-01-01|http://example.com|1|0|1""".encode(
                        "latin-1"
                    )
                )
            with zf.open("ml-100k/u.user", "w") as ofs:
                ofs.write(
                    """1|32|M|0|1690074
    2|4|F|1|1760013
    """.encode()
                )

        loader = MovieLens100KDataManager(ZIPFILE_NAME)
        df = loader.read_interaction()
        movie_info, genres = loader.read_item_info()
        user_info = loader.read_user_info()
        assert df.shape == (2, 4)
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
        np.testing.assert_array_equal(movie_info.index.values, [1, 2])
        np.testing.assert_array_equal(
            movie_info.release_date,
            np.asarray(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]"),
        )
        assert set(genres[genres.movieId == 1].genre) == set(["action"])
        assert set(genres[genres.movieId == 2].genre) == set(["fantasy", "thriller"])

        np.testing.assert_array_equal(user_info.index.values, [1, 2])
        np.testing.assert_array_equal(user_info.gender, ["M", "F"])
    finally:
        os.remove(ZIPFILE_NAME)


def test_ml1m() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")

    try:
        with ZipFile(ZIPFILE_NAME, "w") as zf:
            with zf.open("ml-1m/ratings.dat", "w") as ofs:
                ofs.write(
                    """1::2::5::0
    1::3::5::86400
    """.encode()
                )
            with zf.open("ml-1m/movies.dat", "w") as ofs:
                ofs.write(
                    """1::A fantastic movie (2020)::fantasy|thriller
    1917::Vinni-Pukh(1969)::children
    """.encode(
                        "latin-1"
                    )
                )
            with zf.open("ml-1m/users.dat", "w") as ofs:
                ofs.write(
                    """1::M::32::0::1690074
    2::F::4::1::1760013
    """.encode()
                )

        loader = MovieLens1MDataManager(ZIPFILE_NAME)
        df = loader.read_interaction()
        movie_info = loader.read_item_info()
        user_info = loader.read_user_info()
        assert df.shape == (2, 4)
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
        np.testing.assert_array_equal(movie_info.index.values, [1, 1917])
        np.testing.assert_array_equal(movie_info.release_year, [2020, 1969])
        np.testing.assert_array_equal(user_info.index.values, [1, 2])
        np.testing.assert_array_equal(user_info.gender, ["M", "F"])
    finally:
        os.remove(ZIPFILE_NAME)


def test_ml20m() -> None:
    if sys.platform == "win32":
        pytest.skip("Skip on Windows.")
    try:
        with ZipFile(ZIPFILE_NAME, "w") as zf:
            with zf.open("ml-20m/ratings.csv", "w") as ofs:
                ofs.write(
                    """userId,movieId,rating,timestamp
    1,2,5,0
    1,3,5,86400
    """.encode()
                )
        loader = MovieLens20MDataManager(ZIPFILE_NAME)
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
    finally:
        os.remove(ZIPFILE_NAME)
