from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, UserIndexArray
from irspack.utils._util_cpp import retrieve_recommend_from_score
from irspack.utils.threading import get_n_threads

if TYPE_CHECKING:
    # We should move this module out of "utils".
    from irspack.recommenders import BaseRecommender


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

    def _item_id_list_to_index_list(self, ids: List[Any]) -> List[int]:
        return [self.item_id_to_index[id] for id in ids if id in self.item_id_to_index]

    def _item_id_list_to_array(self, ids: List[Any]) -> np.ndarray:
        return np.asarray(
            self._item_id_list_to_index_list(ids),
            dtype=np.int64,
        )

    def _list_of_user_profile_to_matrix(
        self, users_info: Sequence[Union[List[Any], Dict[Any, float]]]
    ) -> sps.csr_matrix:
        data: List[float] = []
        row: List[int] = []
        col: List[int] = []
        for user_index, user_info in enumerate(users_info):
            if isinstance(user_info, list):
                for iid in user_info:
                    if iid in self.item_id_to_index:
                        data.append(1.0)
                        row.append(user_index)
                        col.append(self.item_id_to_index[iid])
            elif isinstance(user_info, dict):
                for iid, rating in user_info.items():
                    if iid in self.item_id_to_index:
                        data.append(rating)
                        row.append(user_index)
                        col.append(self.item_id_to_index[iid])
        result = sps.csr_matrix(
            (data, (row, col)), shape=(len(users_info), len(self.item_ids))
        )
        return result

    def get_recommendation_for_known_user_id(
        self,
        user_id: Any,
        cutoff: int = 20,
        allowed_item_ids: Optional[List[Any]] = None,
        forbidden_item_ids: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, float]]:
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

    def get_recommendation_for_new_user(
        self,
        item_ids: List[Any],
        cutoff: int = 20,
        allowed_item_ids: Optional[List[Any]] = None,
        forbidden_item_ids: Optional[List[Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """Retrieve recommendation result for a previously unseen user using item ids with which he or she interacted.

        Args:
            item_ids:
                Item IDs the user had an interaction with.
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
        cols = self._item_id_list_to_array(item_ids)
        rows = np.zeros_like(cols)
        data = np.ones(cols.shape, dtype=np.float64)
        X_input = sps.csr_matrix((data, (rows, cols)), shape=(1, len(self.item_ids)))
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
            allowed_item_indices = self._item_id_list_to_array(allowed_item_ids)
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

        allowed_item_indices: List[List[int]] = []
        if allowed_item_ids is not None:
            allowed_item_indices = [
                self._item_id_list_to_index_list(_) for _ in allowed_item_ids
            ]
        forbidden_item_indices: List[List[int]] = []
        if forbidden_item_ids is not None:
            forbidden_item_indices = [
                self._item_id_list_to_index_list(_) for _ in forbidden_item_ids
            ]
        raw_result = retrieve_recommend_from_score(
            score,
            allowed_item_indices,
            forbidden_item_indices,
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
