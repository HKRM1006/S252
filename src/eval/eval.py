import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
MODELS = {
    'RF':   RandomForestClassifier(n_estimators=100, random_state=42),
    'ET':   ExtraTreesClassifier(
                n_estimators=300, random_state=42, class_weight='balanced',
                n_jobs=1,
            ),
    'MLP':  MLPClassifier(
                random_state=42, max_iter=500, early_stopping=True,
                hidden_layer_sizes=(100,), activation='relu', solver='adam',
            ),
    'KNN':  KNeighborsClassifier(n_neighbors=5),
    'LR':   LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
    'XGB':  XGBClassifier(random_state=42, eval_metric='logloss'),
    'LGBM': LGBMClassifier(random_state=42),
}


def _positive_class_score(model, X_test):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, 'decision_function'):
        score = model.decision_function(X_test)
        if np.ndim(score) > 1:
            return score[:, 1]
        return score

    return None


def _safe_curve_metric(metric_fn, y_true, y_score):
    if y_score is None or pd.Series(y_true).nunique() < 2:
        return np.nan
    try:
        return round(metric_fn(y_true, y_score), 4)
    except ValueError:
        return np.nan


def evaluate(X_train, X_test, y_train, y_test,
             models: dict = None, average: str = 'macro') -> pd.DataFrame:
    if models is None:
        models = MODELS

    results = []
    for name, model in models.items():
        start = time.time()
        model = clone(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = _positive_class_score(model, X_test)
        elapsed = round(time.time() - start, 2)

        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Balanced Accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, y_pred,    average=average, zero_division=0), 4),
            'F1-Score':  round(f1_score(y_test, y_pred,        average=average, zero_division=0), 4),
            'MCC':       round(matthews_corrcoef(y_test, y_pred), 4),
            'ROC-AUC':   _safe_curve_metric(roc_auc_score, y_test, y_score),
            'PR-AUC':    _safe_curve_metric(average_precision_score, y_test, y_score),
            'Time (s)':  elapsed,
        })

    return pd.DataFrame(results)
