import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
MODELS = {
    'RF':   RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP':  MLPClassifier(
                random_state=42, max_iter=500, early_stopping=True,
                hidden_layer_sizes=(100,), activation='relu', solver='adam',
            ),
    'KNN':  KNeighborsClassifier(n_neighbors=5),
    'LR':   LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
    'XGB':  XGBClassifier(random_state=42, eval_metric='logloss'),
    'LGBM': LGBMClassifier(random_state=42),
}

def evaluate(X_train, X_test, y_train, y_test,
             models: dict = None, average: str = 'macro') -> pd.DataFrame:
    if models is None:
        models = MODELS

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = round(time.time() - start, 2)

        results.append({
            'Model':     name,
            'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, y_pred,    average=average, zero_division=0), 4),
            'F1-Score':  round(f1_score(y_test, y_pred,        average=average, zero_division=0), 4),
            'Time (s)':  elapsed,
        })

    return pd.DataFrame(results)
