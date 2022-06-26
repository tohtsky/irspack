from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse as sps

from .._threading import get_n_threads
from ..definitions import DenseScoreArray, UserIndexArray
from ._util_cpp import (
    retrieve_recommend_from_score_f32,
    retrieve_recommend_from_score_f64,
)

if TYPE_CHECKING:
    # We should move this module out of "utils".
    from ..recommenders import BaseRecommender


def retrieve_recommend_from_score(
    score: DenseScoreArray,
    allowed_item_indices: List[List[int]],
    cutoff: int,
    n_threads: int,
) -> List[List[Tuple[int, float]]]:
    if score.dtype == np.float32:
        return retrieve_recommend_from_score_f32(
            score, allowed_item_indices, cutoff, n_threads
        )
    elif score.dtype == np.float64:
        return retrieve_recommend_from_score_f64(
            score, allowed_item_indices, cutoff, n_threads
        )
    else:
        raise ValueError("Only float32 or float64 are allowed.")


UserIdType = TypeVar("UserIdType")
ItemIdType = TypeVar("ItemIdType")


class ItemIDMapper(Generic[ItemIdType]):
    r"""A utility class that helps mapping item IDs to indices or vice versa.

    Args:
        item_ids:
            List of item IDs. The ordering of this list should be consistent with
            the item indices of recommenders or score arrays to be used.

    Raises:
        ValueError: When there is a duplicate in item_ids.

    """

    def __init__(
        self,
        item_ids: List[ItemIdType],
    ):
        self.item_ids = item_ids
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(item_ids)}
        if len(self.item_ids) != len(self.item_id_to_index):
            raise ValueError("Duplicates in item_ids.")

    def _check_recommender_n_items(self, rec: "BaseRecommender") -> None:
        if rec.n_items != len(self.item_ids):
            raise ValueError("`n_items` of the recommender is inconsistent.")

    def _check_score_shape(self, score: DenseScoreArray) -> None:
        if score.shape[1] != len(self.item_ids):
            raise ValueError("`score.shape[1]` inconsistent with `len(self.item_ids)`")

    def _item_id_list_to_index_list(self, ids: Iterable[ItemIdType]) -> List[int]:
        return [self.item_id_to_index[id] for id in ids if id in self.item_id_to_index]

    def _user_profile_to_data_col(
        self, profile: Union[List[ItemIdType], Dict[ItemIdType, float]]
    ) -> Tuple[List[float], List[int]]:
        data: List[float]
        cols: List[int]
        # data: np.ndarray
        if isinstance(profile, list):
            cols = self._item_id_list_to_index_list(profile)
            data = [1.0] * len(cols)
        else:
            data = []
            cols = []
            for id, score in profile.items():
                if id in self.item_id_to_index:
                    data.append(score)
                    cols.append(self.item_id_to_index[id])
        return data, cols

    def list_of_user_profile_to_matrix(
        self, users_info: Sequence[Union[List[ItemIdType], Dict[ItemIdType, float]]]
    ) -> sps.csr_matrix:
        r"""Converts users' profiles (interaction histories for the users) into a sparse matrix.

        Args:
            users_info:
                A list of user profiles.
                Each profile should be either the item ids that the user cotacted or a dictionary of item ratings.
                Previously unseen item IDs will be ignored.

        Returns:
            The converted sparse matrix. Each column correspond to `self.items_ids`.
        """
        data: List[float] = []
        indptr: List[int] = [0]
        col: List[int] = []
        indptr_current = 0
        for user_info in users_info:
            data_u, col_u = self._user_profile_to_data_col(user_info)
            data.extend(data_u)
            col.extend(col_u)
            indptr_current += len(col_u)
            indptr.append(indptr_current)
        result = sps.csr_matrix(
            (data, col, indptr), shape=(len(users_info), len(self.item_ids))
        )
        return result

    def recommend_for_new_user(
        self,
        recommender: "BaseRecommender",
        user_profile: Union[List[ItemIdType], Dict[ItemIdType, float]],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        forbidden_item_ids: Optional[List[ItemIdType]] = None,
    ) -> List[Tuple[ItemIdType, float]]:
        r"""Retrieves recommendations for an unknown user by using the user's contact history with the known items.
        Args:
            recommender:
                The recommender for scoring.
            user_profile:
                User's profile given either as a list of item ids the user had a cotact or a item id-rating dict.
                Previously unseen item ID will be ignored.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, recommend the items within this list.
                If ``None``, all known item ids can be recommended (except for those in ``item_ids`` argument).
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, never recommend the items within the list. Defaults to None.

        Returns:
            A List of tuples consisting of ``(item_id, score)``.
        """
        self._check_recommender_n_items(recommender)
        data, cols = self._user_profile_to_data_col(user_profile)
        X_input = sps.csr_matrix(
            (data, cols, [0, len(cols)]), shape=(1, len(self.item_ids))
        )
        score = recommender.get_score_cold_user_remove_seen(X_input)[0]
        return self.score_to_recommended_items(
            score,
            cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
        )

    def recommend_for_new_user_batch(
        self,
        recommender: "BaseRecommender",
        user_profiles: Sequence[Union[List[ItemIdType], Dict[ItemIdType, float]]],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        per_user_allowed_item_ids: Optional[List[List[ItemIdType]]] = None,
        forbidden_item_ids: Optional[List[List[ItemIdType]]] = None,
        n_threads: Optional[int] = None,
    ) -> List[List[Tuple[ItemIdType, float]]]:
        r"""Retrieves recommendations for unknown users by using their contact history with the known items.

        Args:
            recommender:
                The recommender for scoring.
            user_profiles:
                A list of user profiles.
                Each profile should be either the item ids the user had a cotact, or item-rating dict.
                Previously unseen item IDs will be ignored.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, defines "a list of recommendable item IDs".
                Ignored if `per_user_allowed_item_ids` is set.
            per_user_allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``score.shape[0]``.
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, defines "a list of list of forbidden item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``
                Defaults to ``None``.
            n_threads:
                Specifies the number of threads to use for the computation.
                If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
                and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.

        Returns:
            A list of list of tuples consisting of ``(item_id, score)``.
            Each internal list corresponds to the recommender's recommendation output.
        """
        self._check_recommender_n_items(recommender)
        X_input = self.list_of_user_profile_to_matrix(user_profiles)
        score = recommender.get_score_cold_user_remove_seen(X_input)
        return self.score_to_recommended_items_batch(
            score,
            cutoff,
            allowed_item_ids=allowed_item_ids,
            per_user_allowed_item_ids=per_user_allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
            n_threads=get_n_threads(n_threads=n_threads),
        )

    def score_to_recommended_items(
        self,
        score: DenseScoreArray,
        cutoff: int,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        forbidden_item_ids: Optional[List[ItemIdType]] = None,
    ) -> List[Tuple[ItemIdType, float]]:
        self._check_score_shape(score[None, :])

        if allowed_item_ids is not None:
            allowed_item_indices = np.asarray(
                self._item_id_list_to_index_list(allowed_item_ids), dtype=np.int64
            )
            high_score_inds = allowed_item_indices[
                score[allowed_item_indices].argsort()[::-1]
            ]
        else:
            high_score_inds = score.argsort()[::-1]
        recommendations: List[Tuple[ItemIdType, float]] = []
        for i in high_score_inds:
            i_int = int(i)
            score_this = score[i_int]
            item_id = self.item_ids[i_int]
            if np.isinf(score_this):
                continue
            if forbidden_item_ids is not None:
                if item_id in forbidden_item_ids:
                    continue
            recommendations.append((item_id, float(score_this)))
            if len(recommendations) >= cutoff:
                break
        return recommendations

    def score_to_recommended_items_batch(
        self,
        score: DenseScoreArray,
        cutoff: int,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        per_user_allowed_item_ids: Optional[List[List[ItemIdType]]] = None,
        forbidden_item_ids: Optional[List[List[ItemIdType]]] = None,
        n_threads: Optional[int] = None,
    ) -> List[List[Tuple[ItemIdType, float]]]:
        r"""Retrieve recommendation from score array.
        An item with negative infinity score for a user will not be recommended for the user.

        Args:
            score:
                1d numpy ndarray for score.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, defines "a list of recommendable item IDs".
                Ignored if `per_user_allowed_item_ids` is set.
            per_user_allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``score.shape[0]``.
                Defaults to ``None``.
            allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``.
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, defines "a list of list of forbidden item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``
                Defaults to ``None``.

        """
        self._check_score_shape(score)

        if forbidden_item_ids is not None:
            assert len(forbidden_item_ids) == score.shape[0]
        if per_user_allowed_item_ids is not None:
            assert len(per_user_allowed_item_ids) == score.shape[0]

        allowed_item_indices: List[List[int]] = []
        if per_user_allowed_item_ids is not None:
            allowed_item_indices = [
                self._item_id_list_to_index_list(_) for _ in per_user_allowed_item_ids
            ]
        elif allowed_item_ids is not None:
            allowed_item_indices = [self._item_id_list_to_index_list(allowed_item_ids)]
        if forbidden_item_ids is not None:
            for u, forbidden_ids_per_user in enumerate(forbidden_item_ids):
                score[
                    u, self._item_id_list_to_index_list(forbidden_ids_per_user)
                ] = -np.inf

        raw_result = retrieve_recommend_from_score(
            score,
            allowed_item_indices,
            cutoff,
            n_threads=get_n_threads(n_threads),
        )
        return [
            [
                (self.item_ids[item_index], score)
                for item_index, score in user_wise_raw_result
            ]
            for user_wise_raw_result in raw_result
        ]


class IDMapper(Generic[UserIdType, ItemIdType], ItemIDMapper[ItemIdType]):
    r"""A utility class that helps mapping user/item IDs to indices or vice versa.

    Args:
        user_ids:
            List of user IDs. The ordering should be consistent with
            the user indices of recommenders to be used.

        item_ids:
            List of item IDs. The ordering should be consistent with
            the item indices of recommenders or score arrays to be used.

    Raises:
        ValueError: When there is a duplicate in item_ids.
    """

    def __init__(self, user_ids: List[UserIdType], item_ids: List[ItemIdType]):
        super().__init__(item_ids)
        self.user_ids = user_ids
        self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        if len(self.user_ids) != len(self.user_id_to_index):
            raise ValueError("Duplicates in user_ids.")

    def _check_recommender_n_users(self, rec: "BaseRecommender") -> None:
        if rec.n_users != len(self.user_ids):
            raise ValueError("")

    def recommend_for_known_user_id(
        self,
        recommender: "BaseRecommender",
        user_id: UserIdType,
        cutoff: int = 20,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        forbidden_item_ids: Optional[List[ItemIdType]] = None,
    ) -> List[Tuple[ItemIdType, float]]:
        r"""Retrieve recommendation result for a known user.
        Args:
            recommender:
                The recommender for scoring.
            user_id:
                The target user ID.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, recommend the items within this list.
                If ``None``, all known item ids can be recommended (except for those in ``item_ids`` argument).
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, never recommend the items within the list. Defaults to None.

        Raises:
            RuntimeError: When user_id is not in ``self.user_ids``.

        Returns:
            A List of tuples consisting of ``(item_id, score)``.
        """
        self._check_recommender_n_users(recommender)
        if user_id not in self.user_ids:
            raise RuntimeError(f"User with user_id {user_id} not found.")
        user_index: UserIndexArray = np.asarray(
            [self.user_id_to_index[user_id]], dtype=np.int64
        )

        score = recommender.get_score_remove_seen(user_index)[0, :]
        return self.score_to_recommended_items(
            score,
            cutoff=cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
        )

    def recommend_for_known_user_batch(
        self,
        recommender: "BaseRecommender",
        user_ids: List[UserIdType],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[ItemIdType]] = None,
        per_user_allowed_item_ids: Optional[List[List[ItemIdType]]] = None,
        forbidden_item_ids: Optional[List[List[ItemIdType]]] = None,
        n_threads: Optional[int] = None,
    ) -> List[List[Tuple[ItemIdType, float]]]:
        r"""Retrieves recommendation for known users.

        Args:
            recommender:
                The recommender for scoring.
            user_ids:
                A list of user ids.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, defines "a list of recommendable item IDs".
                Ignored if `per_user_allowed_item_ids` is set.
            per_user_allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``score.shape[0]``.
                Defaults to ``None``.

            forbidden_item_ids:
                If not ``None``, defines "a list of list of forbidden item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``
                Defaults to ``None``.
            n_threads:
                Specifies the number of threads to use for the computation.
                If ``None``, the environment variable ``"IRSPACK_NUM_THREADS_DEFAULT"`` will be looked up,
                and if the variable is not set, it will be set to ``os.cpu_count()``. Defaults to None.

        Returns:
            A list of list of tuples consisting of ``(item_id, score)``.
            Each internal list corresponds to the recommender's recommendation output.
        """
        self._check_recommender_n_users(recommender)

        user_indexes: UserIndexArray = np.asarray(
            [self.user_id_to_index[user_id] for user_id in user_ids], dtype=np.int64
        )

        score = recommender.get_score_remove_seen(user_indexes)
        return self.score_to_recommended_items_batch(
            score,
            cutoff=cutoff,
            allowed_item_ids=allowed_item_ids,
            per_user_allowed_item_ids=per_user_allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
            n_threads=get_n_threads(n_threads=n_threads),
        )
