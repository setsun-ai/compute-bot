"""Model zoo with hyperparameter grids, tuned by cross-validation on the training set only."""
from __future__ import annotations

import inspect
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold, cross_val_predict

LINEAR = ("mlr", "pls", "ridge", "lasso", "enet", "logistic", "logistic_l1", "logistic_enet", "plsda")
# interpretability order, used to break near-ties when choosing the best model
SIMPLICITY = ("mlr", "pls", "ridge", "lasso", "enet", "logistic", "logistic_l1", "logistic_enet", "plsda", "knn",
              "dt", "svr", "svm", "rf", "xgb")
CLASS_MAP = {"mlr": "logistic", "ridge": "logistic", "lasso": "logistic_l1", "enet": "logistic_enet", "pls": "plsda",
             "knn": "knn", "dt": "dt", "rf": "rf", "svr": "svm", "xgb": "xgb"}
LOO_MAX = 60  # Q² by leave-one-out up to this many training compounds, 10-fold CV above


class PLSDA(ClassifierMixin, BaseEstimator):
    """PLS discriminant analysis: PLS regression on one-hot classes, the largest predicted column wins."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, x, y):
        from sklearn.cross_decomposition import PLSRegression
        self.classes_ = np.unique(y)
        target = (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)
        self.pls_ = PLSRegression(n_components=min(self.n_components, x.shape[1]), scale=False).fit(x, target)
        return self

    def decision_function(self, x):
        return self.pls_.predict(x)

    def predict_proba(self, x):
        s = np.clip(self.decision_function(x), 0, None) + 1e-9
        return s / s.sum(1, keepdims=True)

    def predict(self, x):
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


def _logistic(l1_ratio=0.0, seed=0, **kw):
    from sklearn.linear_model import LogisticRegression
    new_api = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"
    if new_api:
        solver = "lbfgs" if l1_ratio == 0 else "saga"
        return LogisticRegression(l1_ratio=l1_ratio, solver=solver, class_weight="balanced", max_iter=10000,
                                  random_state=seed, **kw)
    if l1_ratio == 0:
        return LogisticRegression(class_weight="balanced", max_iter=10000, random_state=seed, **kw)
    return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", class_weight="balanced",
                              max_iter=10000, random_state=seed, **kw)


def regression_zoo(name, n_train, k, seed):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    if name == "mlr":
        return LinearRegression(), {}
    if name == "pls":
        return PLSRegression(scale=False), {"n_components": list(range(1, max(1, min(k, 10, n_train - 2)) + 1))}
    if name == "ridge":
        return Ridge(), {"alpha": list(np.logspace(-3, 3, 13))}
    if name == "lasso":
        return Lasso(max_iter=100000), {"alpha": list(np.logspace(-4, 0.5, 10))}
    if name == "enet":
        return ElasticNet(max_iter=100000), {"alpha": list(np.logspace(-4, 0.5, 10)), "l1_ratio": [0.2, 0.5, 0.8]}
    if name == "knn":
        return KNeighborsRegressor(), {"n_neighbors": list(range(1, max(1, min(10, n_train - 2)) + 1)),
                                       "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
    if name == "dt":
        return DecisionTreeRegressor(random_state=seed), {"max_depth": [2, 3, 4, 5, None],
                                                          "min_samples_leaf": [1, 2, 3, 5]}
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1), {
            "max_features": [1.0, "sqrt", 0.5], "min_samples_leaf": [1, 2, 4]}
    if name == "svr":
        return TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler()), {
            "regressor__C": [0.1, 1.0, 10.0, 100.0], "regressor__gamma": ["scale", 0.01, 0.1, 1.0],
            "regressor__epsilon": [0.05, 0.1, 0.3]}
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(subsample=0.8, n_jobs=1, random_state=seed, verbosity=0), {
            "n_estimators": [100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]}
    raise ValueError(name)


def classification_zoo(name, n_train, k, seed):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    cs = list(np.logspace(-2, 3, 6))
    if name == "logistic":
        return _logistic(0.0, seed), {"C": cs}
    if name == "logistic_l1":
        return _logistic(1.0, seed), {"C": cs}
    if name == "logistic_enet":
        return _logistic(0.5, seed), {"C": cs, "l1_ratio": [0.2, 0.5, 0.8]}
    if name == "plsda":
        return PLSDA(), {"n_components": list(range(1, max(1, min(k, 10)) + 1))}
    if name == "knn":
        return KNeighborsClassifier(), {"n_neighbors": list(range(1, max(1, min(15, n_train - 2)) + 1, 2)),
                                        "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
    if name == "dt":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed), {
            "max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 2, 3, 5]}
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1), {
            "max_features": [1.0, "sqrt", 0.5], "min_samples_leaf": [1, 2, 4]}
    if name == "svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed), {
            "C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1, 1.0]}
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(subsample=0.8, n_jobs=1, random_state=seed, verbosity=0), {
            "n_estimators": [100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1]}
    raise ValueError(name)


def tuning_cv(y, task, seed):
    if task == "regression":
        return LeaveOneOut() if len(y) < 15 else KFold(n_splits=5, shuffle=True, random_state=seed)
    _, counts = np.unique(y, return_counts=True)
    return StratifiedKFold(n_splits=max(2, min(5, counts.min())), shuffle=True, random_state=seed)


def validation_cv(y, task, seed):
    """Scheme for Q² / CV metrics of the final models: LOO for small sets, 10-fold above LOO_MAX."""
    if len(y) <= LOO_MAX:
        return LeaveOneOut(), "LOO"
    if task == "regression":
        return KFold(n_splits=10, shuffle=True, random_state=seed), "10-fold"
    _, counts = np.unique(y, return_counts=True)
    return StratifiedKFold(n_splits=max(2, min(10, counts.min())), shuffle=True, random_state=seed), "10-fold"


def _fit_params(name, y, task):
    if task == "classification" and name == "xgb":
        from sklearn.utils.class_weight import compute_sample_weight
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


def tune(name, x, y, task, seed, threads=1):
    """Grid search by CV on the training set; returns (fitted best estimator, best params)."""
    n, k = x.shape
    est, grid = (regression_zoo if task == "regression" else classification_zoo)(name, n, k, seed)
    fit_kw = _fit_params(name, y, task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not grid:
            return est.fit(x, y, **fit_kw), {}
        scoring = "neg_mean_squared_error" if task == "regression" else "balanced_accuracy"
        gs = GridSearchCV(est, grid, cv=tuning_cv(y, task, seed), scoring=scoring, n_jobs=threads, refit=True)
        gs.fit(x, y, **fit_kw)
    return gs.best_estimator_, {k: (float(v) if isinstance(v, np.floating) else v) for k, v in gs.best_params_.items()}


def predict(est, x):
    return np.asarray(est.predict(x)).ravel()


def cv_predict(name, est, x, y, task, seed, threads=1, method="predict"):
    """Cross-validated predictions with the tuned hyperparameters fixed (Q²LOO / CV metrics)."""
    cv, _ = validation_cv(y, task, seed)
    fit_kw = _fit_params(name, y, task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = cross_val_predict(clone(est), x, y, cv=cv, n_jobs=threads, method=method, params=fit_kw or None)
    return np.asarray(out) if method != "predict" else np.asarray(out).ravel()


def refit(name, est, x, y, task):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clone(est).fit(x, y, **_fit_params(name, y, task))
