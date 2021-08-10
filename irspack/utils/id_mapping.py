from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, UserIndexArray
from irspack.utils._util_cpp import (
    retrieve_recommend_from_score_f32,
    retrieve_recommend_from_score_f64,
)
from irspack.utils.threading import get_n_threads

if TYPE_CHECKING:
    # We should move this module out of "utils".
    from irspack.recommenders import BaseRecommender


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


class IDMappedRecommender:
    """A utility class that helps  mapping user/item ids to index, retrieving recommendation score,
    and making a recommendation.

    Args:
        recommender:
            The backend base recommender which transforms user/item ids.
        user_ids:
            user_ids which correspods to the rows of ``recommender.X_train_all``.
        item_ids:
            item_ids which correspods to the columns of ``recommender.X_train_all``.

    Raises:
        ValueError: When recommender and user_ids/item_ids are inconsistent.
        ValueError: When there is a duplicate in user_ids.
        ValueError: When there is a duplicate in item_ids.

    """

    def __init__(
        self, recommender: "BaseRecommender", user_ids: List[Any], item_ids: List[Any]
    ):

        if (recommender.n_users != len(user_ids)) or (
            recommender.n_items != len(item_ids)
        ):
            raise ValueError(
                "The recommender and user/item ids have inconsistent lengths."
            )

        self.recommender = recommender
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_id_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

    def _item_id_list_to_index_list(self, ids: Iterable[Any]) -> List[int]:
        return [self.item_id_to_index[id] for id in ids if id in self.item_id_to_index]

    def _user_profile_to_data_col(
        self, profile: Union[List[Any], Dict[Any, float]]
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

    def _list_of_user_profile_to_matrix(
        self, users_info: Sequence[Union[List[Any], Dict[Any, float]]]
    ) -> sps.csr_matrix:
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

    def get_recommendation_for_known_user_id(
        self,
        user_id: Any,
        cutoff: int = 20,
        allowed_item_ids: Optional[List[Any]] = None,
        forbidden_item_ids: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Retrieve recommendation result for a known user.
        Args:
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
        if user_id not in self.user_ids:
            raise RuntimeError(f"User with user_id {user_id} not found.")
        user_index: UserIndexArray = np.asarray(
            [self.user_id_to_index[user_id]], dtype=np.int64
        )

        score = self.recommender.get_score_remove_seen(user_index)[0, :]
        return self._score_to_recommended_items(
            score,
            cutoff=cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
        )

    def get_recommendation_for_known_user_batch(
        self,
        user_ids: List[Any],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[List[Any]]] = None,
        forbidden_item_ids: Optional[List[List[Any]]] = None,
        n_threads: Optional[int] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Retrieve recommendation result for a list of known users.

        Args:
            user_ids:
                A list of user ids.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``.
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, defines "a list of list of forbidden item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``
                Defaults to ``None``.

        Returns:
            A list of list of tuples consisting of ``(item_id, score)``.
            Each internal list corresponds to the recommender's recommendation output.
        """
        user_indexes: UserIndexArray = np.asarray(
            [self.user_id_to_index[user_id] for user_id in user_ids], dtype=np.int64
        )

        score = self.recommender.get_score_remove_seen(user_indexes)
        return self._score_to_recommended_items_batch(
            score,
            cutoff=cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
            n_threads=get_n_threads(n_threads=n_threads),
        )

    def get_recommendation_for_new_user(
        self,
        user_profile: Union[List[Any], Dict[Any, float]],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[Any]] = None,
        forbidden_item_ids: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Retrieve recommendation result for a previously unseen user using item ids with which he or she interacted.

        Args:
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
        data, cols = self._user_profile_to_data_col(user_profile)
        X_input = sps.csr_matrix(
            (data, cols, [0, len(cols)]), shape=(1, len(self.item_ids))
        )
        score = self.recommender.get_score_cold_user_remove_seen(X_input)[0]
        return self._score_to_recommended_items(
            score,
            cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
        )

    def get_recommendation_for_new_user_batch(
        self,
        user_profiles: Sequence[Union[List[Any], Dict[Any, float]]],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[List[Any]]] = None,
        forbidden_item_ids: Optional[List[List[Any]]] = None,
        n_threads: Optional[int] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Retrieve recommendation result for a previously unseen users using item ids with which they have interacted.

        Args:
            user_profiles:
                A list of user profiles.
                Each profile should be either the item ids the user had a cotact, or item-rating dict.
                Previously unseen item IDs will be ignored.
            cutoff:
                Maximal number of recommendations allowed.
            allowed_item_ids:
                If not ``None``, defines "a list of list of recommendable item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``.
                Defaults to ``None``.
            forbidden_item_ids:
                If not ``None``, defines "a list of list of forbidden item IDs"
                and ``len(allowed_item_ids)`` must be equal to ``len(item_ids)``
                Defaults to ``None``.

        Returns:
            A list of list of tuples consisting of ``(item_id, score)``.
            Each internal list corresponds to the recommender's recommendation output.
        """
        X_input = self._list_of_user_profile_to_matrix(user_profiles)
        score = self.recommender.get_score_cold_user_remove_seen(X_input)
        return self._score_to_recommended_items_batch(
            score,
            cutoff,
            allowed_item_ids=allowed_item_ids,
            forbidden_item_ids=forbidden_item_ids,
            n_threads=get_n_threads(n_threads=n_threads),
        )

    def _score_to_recommended_items(
        self,
        score: DenseScoreArray,
        cutoff: int,
        allowed_item_ids: Optional[List[Any]] = None,
        forbidden_item_ids: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, float]]:
        if allowed_item_ids is not None:
            allowed_item_indices = np.asarray(
                self._item_id_list_to_index_list(allowed_item_ids), dtype=np.int64
            )
            high_score_inds = allowed_item_indices[
                score[allowed_item_indices].argsort()[::-1]
            ]
        else:
            high_score_inds = score.argsort()[::-1]
        recommendations: List[Tuple[Any, float]] = []
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

    def _score_to_recommended_items_batch(
        self,
        score: DenseScoreArray,
        cutoff: int,
        allowed_item_ids: Optional[List[List[Any]]] = None,
        forbidden_item_ids: Optional[List[List[Any]]] = None,
        n_threads: int = 1,
    ) -> List[List[Tuple[Any, float]]]:
        if forbidden_item_ids is not None:
            assert len(forbidden_item_ids) == score.shape[0]
        if allowed_item_ids is not None:
            assert len(allowed_item_ids) == score.shape[0]

        allowed_item_indices: List[List[int]] = []
        if allowed_item_ids is not None:
            allowed_item_indices = [
                self._item_id_list_to_index_list(_) for _ in allowed_item_ids
            ]
        if forbidden_item_ids is not None:
            for u, forbidden_ids_per_user in enumerate(forbidden_item_ids):
                score[
                    u, self._item_id_list_to_index_list(forbidden_ids_per_user)
                ] = -np.inf

        raw_result = retrieve_recommend_from_score(
            score,
            allowed_item_indices,
            cutoff,
            n_threads=n_threads,
        )
        return [
            [
                (self.item_ids[item_index], score)
                for item_index, score in user_wise_raw_result
            ]
            for user_wise_raw_result in raw_result
        ]
