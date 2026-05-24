import argparse
import os
import time
from typing import List, Tuple

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
LAG_STEPS = (1, 2, 3, 6, 12, 24, 48)
ROLL_WINDOWS = (3, 6, 12, 24, 48)

MODELS = {
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'ET': ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        n_jobs=1,
    ),
    'MLP': MLPClassifier(
        random_state=42,
        max_iter=500,
        early_stopping=True,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LR': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
    'XGB': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LGBM': LGBMClassifier(random_state=42),
}


def load_data(data_dir: str = DATA_DIR):
    df_train = pd.read_csv(os.path.join(data_dir, 'df_train.csv'), index_col=0, parse_dates=True)
    df_test = pd.read_csv(os.path.join(data_dir, 'df_test.csv'), index_col=0, parse_dates=True)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0, parse_dates=True).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0, parse_dates=True).squeeze()
    return df_train, df_test, y_train, y_test


def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_norm, X_test_norm


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.abs() + 1e-6)


def extract_weather_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('DataFrame index must be a DatetimeIndex.')

    df_feat = df.copy()
    for col in df.columns:
        shifted = df_feat[col].shift(1)
        for lag in LAG_STEPS:
            df_feat[f'{col}_lag_{lag}h'] = df_feat[col].shift(lag)

        df_feat[f'{col}_delta_1h'] = df_feat[col] - df_feat[col].shift(1)
        df_feat[f'{col}_delta_3h'] = df_feat[col] - df_feat[col].shift(3)
        df_feat[f'{col}_delta_6h'] = df_feat[col] - df_feat[col].shift(6)

        for window in ROLL_WINDOWS:
            rolling = shifted.rolling(window=window, min_periods=max(2, window // 2))
            df_feat[f'{col}_roll_{window}h_mean'] = rolling.mean()
            df_feat[f'{col}_roll_{window}h_std'] = rolling.std()
            df_feat[f'{col}_roll_{window}h_min'] = rolling.min()
            df_feat[f'{col}_roll_{window}h_max'] = rolling.max()
            df_feat[f'{col}_roll_{window}h_range'] = rolling.max() - rolling.min()
            df_feat[f'{col}_vs_roll_{window}h_mean'] = df_feat[col] - rolling.mean()

    if 'Precipitation' in df_feat.columns:
        precip = df_feat['Precipitation']
        wet = (precip.shift(1) > 0).astype(int)
        dry = 1 - wet
        wet_blocks = (wet != wet.shift()).cumsum()
        dry_blocks = (dry != dry.shift()).cumsum()

        df_feat['rain_now'] = (precip > 0).astype(int)
        df_feat['rain_prev_1h'] = wet
        df_feat['wet_streak_v2'] = wet.groupby(wet_blocks).cumsum()
        df_feat['dry_streak_v2'] = dry.groupby(dry_blocks).cumsum()

        for window in ROLL_WINDOWS:
            shifted_precip = precip.shift(1)
            df_feat[f'precip_sum_{window}h'] = shifted_precip.rolling(window, min_periods=1).sum()
            df_feat[f'precip_max_{window}h'] = shifted_precip.rolling(window, min_periods=1).max()
            df_feat[f'wet_ratio_{window}h_v2'] = wet.rolling(window, min_periods=1).mean()

        df_feat['rain_memory_balance_6_24h'] = (
            df_feat['precip_sum_6h'] - df_feat['precip_sum_24h'] / 4.0
        )

    if {'Temperature', 'Dew_Point'}.issubset(df_feat.columns):
        spread = df_feat['Temperature'] - df_feat['Dew_Point']
        df_feat['temp_dewpoint_spread'] = spread
        df_feat['temp_dewpoint_spread_lag_1h'] = spread.shift(1)
        df_feat['temp_dewpoint_spread_delta_3h'] = spread - spread.shift(3)

    if {'Humidity', 'Temperature'}.issubset(df_feat.columns):
        df_feat['humidity_temp_interaction'] = df_feat['Humidity'] * df_feat['Temperature']
    if {'Humidity', 'Dew_Point'}.issubset(df_feat.columns):
        df_feat['humidity_dewpoint_interaction'] = df_feat['Humidity'] * df_feat['Dew_Point']
    if {'Humidity', 'Pressure'}.issubset(df_feat.columns):
        df_feat['humidity_pressure_interaction'] = df_feat['Humidity'] * df_feat['Pressure']

    if 'Pressure' in df_feat.columns:
        df_feat['pressure_tendency_3h'] = df_feat['Pressure'] - df_feat['Pressure'].shift(3)
        df_feat['pressure_tendency_6h'] = df_feat['Pressure'] - df_feat['Pressure'].shift(6)
        df_feat['pressure_tendency_ratio_6_24h'] = _safe_ratio(
            df_feat['pressure_tendency_6h'],
            df_feat['Pressure'] - df_feat['Pressure'].shift(24),
        )

    if {'Wind_Gust', 'Wind_Speed'}.issubset(df_feat.columns):
        df_feat['gust_speed_gap'] = df_feat['Wind_Gust'] - df_feat['Wind_Speed']
        df_feat['gust_speed_ratio'] = _safe_ratio(df_feat['Wind_Gust'], df_feat['Wind_Speed'])

    if 'Wind_Speed' in df_feat.columns:
        df_feat['wind_speed_power2'] = df_feat['Wind_Speed'] ** 2
        df_feat['wind_speed_power3'] = df_feat['Wind_Speed'] ** 3

    hours = df_feat.index.hour
    months = df_feat.index.month
    dayofyear = df_feat.index.dayofyear
    df_feat['hour_sin_v2'] = np.sin(2 * np.pi * hours / 24)
    df_feat['hour_cos_v2'] = np.cos(2 * np.pi * hours / 24)
    df_feat['month_sin_v2'] = np.sin(2 * np.pi * months / 12)
    df_feat['month_cos_v2'] = np.cos(2 * np.pi * months / 12)
    df_feat['dayofyear_sin_v2'] = np.sin(2 * np.pi * dayofyear / 365.25)
    df_feat['dayofyear_cos_v2'] = np.cos(2 * np.pi * dayofyear / 365.25)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    return df_feat.dropna()


def _prepare_target(df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    y_shifted = y.shift(-1)
    return df.iloc[:-1].copy(), y_shifted.iloc[:-1].copy()


def _select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    num_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    variance = VarianceThreshold(threshold=0.0)
    X_train_var = pd.DataFrame(
        variance.fit_transform(X_train_imp),
        columns=X_train_imp.columns[variance.get_support()],
        index=X_train_imp.index,
    )
    X_test_var = pd.DataFrame(
        variance.transform(X_test_imp),
        columns=X_train_var.columns,
        index=X_test_imp.index,
    )

    k = min(num_features, X_train_var.shape[1])
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
        k=k,
    )
    X_train_sel = pd.DataFrame(
        selector.fit_transform(X_train_var, y_train),
        columns=X_train_var.columns[selector.get_support()],
        index=X_train_var.index,
    )
    X_test_sel = pd.DataFrame(
        selector.transform(X_test_var),
        columns=X_train_sel.columns,
        index=X_test_var.index,
    )
    return X_train_sel, X_test_sel, X_train_sel.columns.tolist()


def custom_v2_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_features: int = 60,
):
    train_base, y_train_next = _prepare_target(df_train, y_train)
    test_base, y_test_next = _prepare_target(df_test, y_test)

    X_train_feat = extract_weather_memory_features(train_base)
    X_test_feat = extract_weather_memory_features(test_base)

    y_train_aligned = y_train_next.loc[X_train_feat.index]
    y_test_aligned = y_test_next.loc[X_test_feat.index]

    X_train_sel, X_test_sel, selected_features = _select_features(
        X_train_feat,
        X_test_feat,
        y_train_aligned,
        num_features=num_features,
    )

    X_train_norm, X_test_norm = normalize_features(X_train_sel, X_test_sel)
    X_train_res, y_train_res = apply_smote(X_train_norm, y_train_aligned)
    return X_train_res, X_test_norm, y_train_res, y_test_aligned, selected_features


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


def evaluate(X_train, X_test, y_train, y_test, models: dict = None, average: str = 'macro'):
    models = models or MODELS
    results = []
    for name, model in models.items():
        start = time.time()
        model = clone(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = _positive_class_score(model, X_test)
        elapsed = round(time.time() - start, 2)

        results.append({
            'Feature': 'Custom_V2',
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Balanced Accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4),
            'ROC-AUC': _safe_curve_metric(roc_auc_score, y_test, y_score),
            'PR-AUC': _safe_curve_metric(average_precision_score, y_test, y_score),
            'Time (s)': elapsed,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Run Custom_V2 feature set and evaluation.')
    parser.add_argument(
        '--data_dir',
        default=DATA_DIR,
        help='Folder containing df_train.csv, df_test.csv, y_train.csv, y_test.csv.',
    )
    parser.add_argument(
        '--num_features',
        type=int,
        default=60,
        help='Number of mutual-information selected features.',
    )
    parser.add_argument(
        '--output',
        default='my_custom_v2_results.csv',
        help='Output CSV path for evaluation results.',
    )
    args = parser.parse_args()

    df_train, df_test, y_train, y_test = load_data(args.data_dir)
    X_train, X_test, y_train_res, y_test_res, selected_features = custom_v2_pipeline(
        df_train,
        df_test,
        y_train,
        y_test,
        num_features=args.num_features,
    )

    print(f'\nSelected {len(selected_features)} features:')
    for idx, feature in enumerate(selected_features, 1):
        print(f'{idx:02d}. {feature}')

    results = evaluate(X_train, X_test, y_train_res, y_test_res)
    results.to_csv(args.output, index=False)

    print('\nCustom_V2 results:')
    print(results.to_string(index=False))
    print(f'\nSaved results to {args.output}')


if __name__ == '__main__':
    main()
