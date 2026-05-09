"""Random Forest and XGBoost wrappers with a uniform fit/predict API."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@dataclass
class FittedModel:
    name: str
    predict: Any            # callable: X -> labels
    predict_proba: Any      # callable: X -> probs
    train_seconds: float
    n_params: int
    estimator: Any = None   # underlying sklearn / xgboost model
    preprocess: Any = None  # optional StandardScaler etc.


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                      params: dict, seed: int) -> FittedModel:
    import time
    t0 = time.time()
    clf = RandomForestClassifier(random_state=seed, **params)
    clf.fit(X_train, y_train)
    dt = time.time() - t0
    n_params = sum(t.tree_.node_count for t in clf.estimators_)
    return FittedModel(
        name="RandomForest",
        predict=clf.predict,
        predict_proba=clf.predict_proba,
        train_seconds=dt,
        n_params=int(n_params),
        estimator=clf,
    )


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                params: dict, seed: int, num_classes: int) -> FittedModel:
    import time
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    t0 = time.time()
    n_estimators = params.pop("n_estimators", 800)
    early = params.pop("early_stopping_rounds", 50)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=early,
        random_state=seed,
        **params,
    )
    model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    dt = time.time() - t0

    def _predict(X):
        return model.predict(scaler.transform(X).astype(np.float32))

    def _predict_proba(X):
        return model.predict_proba(scaler.transform(X).astype(np.float32))

    n_params = int(model.get_booster().num_boosted_rounds())
    return FittedModel(name="XGBoost",
                       predict=_predict, predict_proba=_predict_proba,
                       train_seconds=dt, n_params=n_params,
                       estimator=model, preprocess=scaler)
