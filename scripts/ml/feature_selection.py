"""Small train-fold-only feature-selector interfaces for ML experiments.

Selectors receive already scaled training predictors and labels, and return
column positions relative to the supplied predictor schema.  They deliberately
do not accept test data: the evaluator owns the outer-CV boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# Frozen for C1 reproducibility.  Complexity predictors are all continuous
# summaries, including after fold-local standardization.
MRMR_MI_PARAMETERS = {
    "n_neighbors": 3,
    "random_state": 42,
    "discrete_features": False,
}


class FeatureSelector(Protocol):
    """Minimal selector contract used by the grouped-CV evaluator."""

    requested_n_features: int
    selected_indices_: np.ndarray
    actual_n_features_: int
    fit_n_rows_: int

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "FeatureSelector": ...


@dataclass
class MRMRSelector:
    """Greedy mutual-information-difference (MID) mRMR selector.

    Relevance is ``I(feature; class)``.  At each subsequent step the candidate
    maximizing relevance minus its mean ``I(candidate; selected feature)`` is
    retained.  Both quantities use scikit-learn's k-nearest-neighbour mutual
    information estimators; no additional package is required.
    """

    requested_n_features: int

    def __post_init__(self) -> None:
        try:
            is_positive_integer = (
                not isinstance(self.requested_n_features, bool)
                and int(self.requested_n_features) == self.requested_n_features
                and self.requested_n_features >= 1
            )
        except (TypeError, ValueError, OverflowError):
            is_positive_integer = False
        if not is_positive_integer:
            raise ValueError("requested_n_features must be a positive integer")
        self.requested_n_features = int(self.requested_n_features)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "MRMRSelector":
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=int)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("mRMR requires a non-empty two-dimensional training matrix")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("mRMR labels must align with training rows")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("mRMR requires both classes in the training fold")
        if not np.isfinite(x).all():
            raise ValueError("mRMR training predictors must be finite")

        actual = min(int(self.requested_n_features), x.shape[1])
        relevance = mutual_info_classif(x, y, **MRMR_MI_PARAMETERS)
        remaining = list(range(x.shape[1]))
        selected: list[int] = []
        redundancy = np.zeros(x.shape[1], dtype=float)
        while remaining and len(selected) < actual:
            if selected:
                # MI is estimated once per newly selected feature, then averaged
                # over selected features through the accumulated redundancy.
                newest = x[:, selected[-1]]
                candidate_columns = np.asarray(remaining, dtype=int)
                redundancy[candidate_columns] += mutual_info_regression(
                    x[:, candidate_columns], newest, **MRMR_MI_PARAMETERS
                )
                scores = relevance[candidate_columns] - redundancy[candidate_columns] / len(selected)
            else:
                candidate_columns = np.asarray(remaining, dtype=int)
                scores = relevance[candidate_columns]
            # np.argmax is deterministic and resolves ties by original schema order.
            chosen = int(candidate_columns[int(np.argmax(scores))])
            selected.append(chosen)
            remaining.remove(chosen)

        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.actual_n_features_ = len(selected)
        self.n_features_in_ = x.shape[1]
        self.fit_n_rows_ = x.shape[0]
        return self
