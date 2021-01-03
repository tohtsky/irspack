from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np

from irspack.dataset.movielens import MovieLens1MDataManager


def test_ml1m() -> None:
    fp = NamedTemporaryFile("wb")
    fp.name
    with ZipFile(fp.name, "w") as zf:
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

    loader = MovieLens1MDataManager(fp.name)
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
