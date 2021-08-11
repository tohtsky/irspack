from .amazon_music import AmazonMusicDataManager
from .citeulike import CiteULikeADataManager
from .movielens import (
    MovieLens1MDataManager,
    MovieLens20MDataManager,
    MovieLens100KDataManager,
)

__all__ = [
    "MovieLens100KDataManager",
    "MovieLens1MDataManager",
    "MovieLens20MDataManager",
    "CiteULikeADataManager",
    "AmazonMusicDataManager",
]
