from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer

from src.eval.eval import evaluate
from src.util import DATA_DIR, apply_smote, load_data, normalize_features


LAG_STEPS = (1, 2, 3, 6, 12, 24, 48)
ROLL_WINDOWS = (3, 6, 12, 24, 48)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.abs() + 1e-6)


def extract_weather_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    df_feat = df.copy()
    base_cols = df.columns.tolist()

    for col in base_cols:
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


def _prepare_target(
    df: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
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

    selected_features = X_train_sel.columns.tolist()
    return X_train_sel, X_test_sel, selected_features


def custom_v2_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_features: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    train_base, y_train_next = _prepare_target(df_train, y_train)
    test_base, y_test_next = _prepare_target(df_test, y_test)

    X_train_feat = extract_weather_memory_features(train_base)
    X_test_feat = extract_weather_memory_features(test_base)

    y_train_aligned = y_train_next.loc[X_train_feat.index]
    y_test_aligned = y_test_next.loc[X_test_feat.index]

    if isinstance(y_train_aligned, pd.DataFrame):
        y_train_aligned = y_train_aligned.iloc[:, 0]
    if isinstance(y_test_aligned, pd.DataFrame):
        y_test_aligned = y_test_aligned.iloc[:, 0]

    X_train_sel, X_test_sel, selected_features = _select_features(
        X_train_feat,
        X_test_feat,
        y_train_aligned,
        num_features=num_features,
    )

    X_train_norm, X_test_norm = normalize_features(X_train_sel, X_test_sel)
    X_train_res, y_train_res = apply_smote(X_train_norm, y_train_aligned)
    return X_train_res, X_test_norm, y_train_res, y_test_aligned, selected_features


def run_custom_v2(data_dir: str = None, num_features: int = 60) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    df_train, df_test, y_train, y_test = load_data(data_dir)

    print("\n=== CUSTOM_V2 PIPELINE ====")
    X_train, X_test, y_train_res, y_test_res, selected_features = custom_v2_pipeline(
        df_train,
        df_test,
        y_train,
        y_test,
        num_features=num_features,
    )

    print(f"\n>>> TOP {len(selected_features)} FEATURES SELECTED BY MUTUAL INFORMATION:")
    for idx, feat in enumerate(selected_features, 1):
        print(f"  {idx:02d}. {feat}")
    print("-" * 40)

    results = evaluate(X_train, X_test, y_train_res, y_test_res)
    results.insert(0, 'Feature', 'Custom_V2')
    print(results.to_string(index=False))

    return results
