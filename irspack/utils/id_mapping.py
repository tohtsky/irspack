from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps

from irspack.definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
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
        self, recommender: BaseRecommender, user_ids: List[Any], item_ids: List[Any]
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

    def _item_id_list_to_array(self, ids: List[Any]) -> np.ndarray:
        return np.sort(
            np.asarray(
                [
                    self.item_id_to_index[id]
                    for id in ids
                    if id in self.item_id_to_index
                ],
                dtype=np.int64,
            )
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
