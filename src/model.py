from typing import Self

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def adaboost(max_depth: int = 1, **params) -> AdaBoostClassifier:
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps); `params` go to
    AdaBoostClassifier. The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def oblique_random_forest(
    n_estimators: int = 100, 
    n_components: int = 10, 
    density="auto",
    random_state=None, 
    n_jobs=None, 
    **tree_params
) -> BaggingClassifier:
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample, on `n_components`
    random sparse linear combinations of the features (a new set per tree), so its splits are
    hyperplanes. `tree_params` go to every tree (`max_features` defaults to "sqrt"). Tuned as
    `model__estimator__tree__<param>` and `model__estimator__project__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)


_MODELS = {
    "dummy": DummyClassifier, # always predicts the most frequent class
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    selected_model = model_config["selected_model"]

    params = model_config.get(selected_model, {})
    return _MODELS[selected_model](**params)


def build_pipeline(
    preprocessing_config: dict, 
    model_config: dict, 
    selection_config: dict = None,
    anomaly_config: dict = None,
    memory=None
) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing, feature selection
    (if `selection_config["enabled"]`) and model as one Pipeline, so every learned step is
    re-fit on the training part of each fold. Step names address hyperparameters from outside
    (`model__max_depth`, `select__min_votes`). `memory` (a folder) caches the fitted
    preprocessing and selection steps: a fit on the same rows with the same settings is
    reused instead of repeated, so tuning the model doesn't refit them in every trial.
    """ # TODO: Update this docstring
    steps = [("prep", build_preprocessor(preprocessing_config))]

    if anomaly_config and anomaly_config.get("enabled", False):
        steps.append(("anomaly", AnomalyScore(
            method=anomaly_config["method"],
            n_neighbors=anomaly_config.get("n_neighbours", 20),
            random_state=anomaly_config.get("random_state")
        )))

    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", VoteSelector(
            min_votes=selection_config.get("min_votes", 2),
            random_state=selection_config.get("random_state")
        )))
        
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training
    rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised logistic regression for features it doesn't shrink to zero.
    """
    def __init__(
        self, 
        min_votes: int = 2, 
        lasso_C: float = 0.01,
        random_state: int = None
    ) -> None:
        self.min_votes = min_votes
        self.lasso_C = lasso_C
        self.random_state = random_state

    def fit(self, X, y) -> Self:
        forest = RandomForestClassifier(
            n_estimators=200, 
            min_samples_leaf=20, 
            random_state=self.random_state, 
            n_jobs=1
        ).fit(X, y)
        
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)

        lasso = LogisticRegression(
            l1_ratio=1.0, 
            C=self.lasso_C, 
            solver="liblinear"
        ).fit(X, y)

        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_


class AnomalyScore(TransformerMixin, BaseEstimator):
    """
    Appends one column, `anomaly_score`, to the preprocessed features: how unusual each row is
    compared with the training rows (higher = more unusual). `method` is "isolation_forest"
    (the score is the theory's s(x), about 0.5 for ordinary rows and close to 1 for anomalies)
    or "lof" (the Local Outlier Factor, about 1 for ordinary rows; `n_neighbors` should be
    larger than the largest group of identical rows). The detector is fitted on the training
    rows only and then scores any rows, so it runs unchanged in CV and on data to predict.
    """
    def __init__(
        self, 
        method: str = "isolation_forest", 
        n_neighbors: int = 20, 
        random_state=None
    ) -> None:
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None) -> Self:
        if self.method == "isolation_forest":
            self.detector_ = IsolationForest(random_state=self.random_state)
        elif self.method == "lof":
            self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        else:
            raise ValueError(f"Unknown anomaly method: {self.method}. Options: isolation_forest, lof")
        
        self.detector_.fit(X)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def fit_transform(self, X, y=None):
        """Fits on X and scores it. LOF scores its own training rows without counting each row as its own neighbour."""
        self.fit(X)
        if self.method == "lof":
            return np.column_stack([X, -self.detector_.negative_outlier_factor_])
        return self.transform(X)

    def transform(self, X):
        return np.column_stack([X, -self.detector_.score_samples(X)])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.append(np.asarray(input_features, dtype=object), "anomaly_score")
