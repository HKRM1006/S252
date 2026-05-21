"""
Custom Pipeline: Weather Feature Engineering Module for S252 Project
Description: A custom pipeline for extracting temporal, statistical, 
and domain-specific features from weather time-series data.
"""
from src.util import DATA_DIR
from src.util import load_data, apply_smote
from src.eval.eval import evaluate

import logging
import itertools
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import ccf

from mrmr import mrmr_classif

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_ccf_lags(
    X: pd.DataFrame, 
    y: pd.Series, 
    features: List[str],
    max_lags: int = 24, 
    top_k: int = 3
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Tính toán nhanh Cross-Correlation (CCF) và trả về các lag tối ưu.
    Hàm này đã loại bỏ phần đồ họa để tối ưu tốc độ phản hồi.

    Args:
        X (pd.DataFrame): DataFrame chứa các đặc trưng huấn luyện.
        y (pd.Series): Chuỗi thời gian của biến mục tiêu (Target).
        features (List[str]): Danh sách các đặc trưng cần quét lag.
        max_lags (int): Số bước trễ (giờ) tối đa cần kiểm tra.
        top_k (int): Số lượng lag mạnh nhất cần lấy cho mỗi biến.

    Returns:
        Dict[str, List[Tuple[int, float]]]: Cấu hình lag phục vụ cho khâu trích xuất đặc trưng.
    """
    top_lags_dict = {}

    for col in features:
        if col not in X.columns:
            logger.warning(f"Feature '{col}' not found in X. Skipping.")
            continue
            
        full_ccf = ccf(X[col].values, y.values, adjusted=False)
      
        lags = np.arange(0, min(max_lags + 1, len(full_ccf)))
        corrs = full_ccf[lags]
        corrs = np.nan_to_num(corrs)

        abs_corrs = np.abs(corrs)
        sorted_indices = np.argsort(abs_corrs)[::-1]
        top_indices = sorted_indices[:top_k]
  
        top_lags = [(int(lags[idx]), float(corrs[idx])) for idx in top_indices]
        top_lags_dict[col] = top_lags

    # for feat, lags_corrs in top_lags_dict.items():
    #     lag_str = ", ".join([f"{lag}h ({corr:.3f})" for lag, corr in lags_corrs])
    #     print(f"{feat:20} | {lag_str}")
    # print("-" * 60)

    return top_lags_dict

def extract_change_features(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Extracts change features (first-order differences) for specified columns in a time-series DataFrame."""
    df_feat = df.copy()
    
    missing_cols = [col for col in target_cols if col not in df_feat.columns]
    if missing_cols:
        logger.error(f"Missing columns for diff extraction: {missing_cols}")
        raise KeyError(f"Columns not found: {missing_cols}")

    for col in target_cols:
        df_feat[f'{col}_diff'] = df_feat[col].diff()
        
    return df_feat

def extract_ewma(df: pd.DataFrame, target_cols: List[str], span: int = 24) -> pd.DataFrame:
    """Extracts Exponentially Weighted Moving Average (EWMA) features for specified columns in a time-series DataFrame."""
    df_feat = df.copy()
    
    missing_cols = [col for col in target_cols if col not in df_feat.columns]
    if missing_cols:
        logger.error(f"Missing columns for EWMA extraction: {missing_cols}")
        raise KeyError(f"Columns not found: {missing_cols}")

    for col in target_cols:
        df_feat[f'{col}_ewma'] = df_feat[col].shift(1).ewm(span=span).mean()

    return df_feat

def extract_multi_scale_local_stats(df: pd.DataFrame, target_cols: List[str], window_sizes: List[int]) -> pd.DataFrame:
    """Extracts multi-scale local statistical features (mean, std, skewness, kurtosis) for specified columns in a time-series DataFrame."""
    df_feat = df.copy()
    for col in target_cols:
        if col not in df_feat.columns:
            raise ValueError(f"Column '{col}' does not exist in the dataframe.")
            
        for w in sorted(window_sizes):
            if w <= 0:
                continue
            
            r = df_feat[col].shift(1).rolling(window=w, min_periods=1)
            df_feat[f'{col}_roll_{w}_mean'] = r.mean()
            
            if w >= 2:
                df_feat[f'{col}_roll_{w}_std'] = r.std()
            if w >= 3:
                df_feat[f'{col}_roll_{w}_skew'] = r.skew()
            if w >= 4:
                df_feat[f'{col}_roll_{w}_kurt'] = r.kurt()
    return df_feat

def extract_advanced_stats(df: pd.DataFrame, target_cols: List[str], window_zscore: int = 24, window_slope: int = 24) -> pd.DataFrame:
    """Extracts advanced statistical features: Rolling Range, IQR, Z-Score, and Trend Slope."""
    df_feat = df.copy()

    missing_cols = [col for col in target_cols if col not in df_feat.columns]
    if missing_cols:
        logger.error(f"Missing columns for advanced stat extraction: {missing_cols}")
        raise KeyError(f"Columns not found: {missing_cols}")

    for col in target_cols:
        series_shifted = df_feat[col].shift(1)
        r = series_shifted.rolling(window=window_slope)
        
        df_feat[f'{col}_range_{window_slope}h'] = r.max() - r.min()
        df_feat[f'{col}_iqr_{window_slope}h'] = r.quantile(0.75) - r.quantile(0.25)
        
        r_24 = series_shifted.rolling(window=window_zscore)
        df_feat[f'{col}_zscore_24h'] = (series_shifted - r_24.mean()) / (r_24.std() + 1e-6)
        
        df_feat[f'{col}_slope_{window_slope}h'] = (series_shifted - series_shifted.shift(window_slope)) / window_slope
    return df_feat

# =============================================================================
# 3. TEMPORAL & PATTERN FEATURES
# =============================================================================

def extract_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal patterns based on cyclical time features."""
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
        
    hours = df_feat.index.hour
    df_feat['sine_hour'] = np.sin(2 * np.pi * hours / 24)
    df_feat['cosine_hour'] = np.cos(2 * np.pi * hours / 24)
    
    days = df_feat.index.dayofyear
    df_feat['sine_day'] = np.sin(2 * np.pi * days / 365.25)
    df_feat['cosine_day'] = np.cos(2 * np.pi * days / 365.25)
    return df_feat


def extract_temporal_memories(df: pd.DataFrame, lag_config: Dict[str, List[Tuple[int, float]]]) -> pd.DataFrame:
    """Extracts lag features based on a provided configuration dictionary specifying columns and their corresponding lag tuples."""
    df_feat = df.copy()
    added_features = 0
    
    for col, lag_list in lag_config.items():
        if col not in df_feat.columns:
            logger.warning(f"Feature '{col}' not found in columns. Skipping lag extraction.")
            continue
            
        if isinstance(lag_list, list):
            for lag_tuple in lag_list:
                try:
                    actual_lag = int(lag_tuple[0])
                    if actual_lag <= 0:
                        continue
                        
                    feature_name = f"{col}_lag_{actual_lag}"
                    if feature_name not in df_feat.columns:
                        df_feat[feature_name] = df_feat[col].shift(actual_lag)
                        added_features += 1
                except (TypeError, ValueError, IndexError):
                    logger.error(f"Could not parse lag configuration tuple for '{col}'.")
                    continue
    logger.info(f"Successfully appended {added_features} customized lag features.")
    return df_feat

# =============================================================================
# 4. DOMAIN-SPECIFIC & PRECIPITATION FEATURES (RAINFALL SPECIAL)
# =============================================================================

def extract_streak_features(df: pd.DataFrame, precip_col: str = 'Precipitation', threshold: float = 0.0) -> pd.DataFrame:
    """Extracts streak features for precipitation data, including wet/dry streak lengths, log-transformed streaks, and wet ratio features.
    """
    df_feat = df.copy()
    if precip_col not in df_feat.columns:
        return df_feat
        
    series = df_feat[precip_col].shift(1)
    
    is_wet = (series > threshold).astype(int)
    
    blocks = (is_wet != is_wet.shift()).cumsum()
    
    df_feat['wet_streak_hours'] = is_wet.groupby(blocks).cumsum()
    df_feat['dry_streak_hours'] = (1 - is_wet).groupby(blocks).cumsum()
    
    df_feat['wet_streak_log'] = np.log1p(df_feat['wet_streak_hours'])
    df_feat['dry_streak_log'] = np.log1p(df_feat['dry_streak_hours'])
    
    df_feat['wet_ratio_24h'] = is_wet.rolling(24).mean()
    df_feat['wet_ratio_6h'] = is_wet.rolling(6).mean()
    
    return df_feat

def pairwise_prod(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Extracts pairwise product features between two specified columns."""
    df_feat = df.copy()
    if col1 in df_feat.columns and col2 in df_feat.columns:
        df_feat[f'{col1}_x_{col2}'] = df_feat[col1] * df_feat[col2]
    return df_feat

def pairwise_diff_prod(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Extracts pairwise product features between the first-order differences (diff) 
    of specified columns to capture non-linear trend interactions.
    """
    df_feat = df.copy()
    
    # Tạo danh sách các cột diff thực tế đang có trong DataFrame
    diff_cols = [f'{col}_diff' for col in target_cols if f'{col}_diff' in df_feat.columns]
    
    if len(diff_cols) < 2:
        logger.warning("Not enough diff columns found to generate pairwise diff products.")
        return df_feat

    for col1, col2 in itertools.combinations(diff_cols, 2):
        new_col_name = f'{col1}_x_{col2}'
        df_feat[new_col_name] = df_feat[col1] * df_feat[col2]
        
    return df_feat

def custom_features_extraction(df: pd.DataFrame, columns: List[str], lags_config: Dict[str, List[Tuple[int, float]]]) -> pd.DataFrame:
    """
    Performs comprehensive feature extraction for weather time-series data, including change features, temporal patterns, temporal memories, advanced statistics, and domain-specific features for precipitation.
    """
    if df.isna().any().any():
        raise ValueError("Input data contains NaNs.")
        
    logger.info(f"Starting Lean Pipeline. Shape: {df.shape}")
    df_feat = df.copy()
    
    try:
        df_feat = extract_change_features(df_feat, columns)
        df_feat = pairwise_diff_prod(df_feat, columns)
        df_feat = extract_temporal_patterns(df_feat)
        
        df_feat = extract_temporal_memories(df_feat, lags_config)
        df_feat = extract_ewma(df_feat, columns)

        df_feat = extract_multi_scale_local_stats(df_feat, columns, [3, 6, 12])
        df_feat = extract_advanced_stats(df_feat, columns)

        df_feat = extract_streak_features(df_feat, precip_col='Precipitation')
        
        df_feat = df_feat.dropna()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise e
        
    logger.info(f"Lean Pipeline finished successfully! Output shape: {df_feat.shape}")
    return df_feat

def custom_pipeline(df_train: pd.DataFrame, df_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, num_features: int=30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Runs the custom feature extraction pipeline on the training and testing datasets."""
    y_train_shifted = y_train.shift(-1)
    y_test_shifted = y_test.shift(-1)

    df_train_clean = df_train.iloc[:-1]
    y_train_clean = y_train_shifted.iloc[:-1]

    df_test_clean = df_test.iloc[:-1]
    y_test_clean = y_test_shifted.iloc[:-1]

    features = df_train.columns.tolist()
    
    lag_config = get_ccf_lags(X=df_train_clean, y=y_train_clean, features=features, max_lags=24, top_k=3)
    
    X_train_feat = custom_features_extraction(df=df_train_clean, columns=features, lags_config=lag_config)
    X_test_feat = custom_features_extraction(df=df_test_clean, columns=features, lags_config=lag_config)

    y_train_res = y_train_clean.loc[X_train_feat.index]
    y_test_res = y_test_clean.loc[X_test_feat.index]

    if isinstance(y_train_res, pd.DataFrame):
        y_train_res = y_train_res.iloc[:, 0]
    if isinstance(y_test_res, pd.DataFrame):
        y_test_res = y_test_res.iloc[:, 0]

    selected_features = mrmr_classif(
        X= X_train_feat,
        y=y_train_res,
        K=num_features
    )

    X_train_mrmr = X_train_feat[selected_features]
    X_test_mrmr = X_test_feat[selected_features]
    
    return X_train_mrmr, X_test_mrmr, y_train_res, y_test_res

def run_custom(data_dir: str = None) -> dict:
    data_dir = data_dir or DATA_DIR
    df_train, df_test, y_train, y_test = load_data(data_dir)

    print("\n=== CUSTOM PIPELINE ====")
    X_train, X_test, y_train_res, y_test_res = custom_pipeline(df_train, df_test, y_train, y_test, num_features=20)
    results = evaluate(X_train, X_test, y_train_res, y_test_res)
    results.insert(0, 'Feature', 'Custom')

    print(results.to_string(index=False))
 
    return results