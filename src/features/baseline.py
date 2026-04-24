"""
src/features/baseline.py
========================
Baseline pipeline: TSFRESH / TSFEL feature extraction
trên dữ liệu đã lưu ở data/
"""

import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import tsfel
from src.util import DATA_DIR
from src.util import load_data, apply_smote
from src.eval.eval import evaluate

WINDOW_SIZE = 12


# ==============================================================================
# TSFRESH
# ==============================================================================
def _make_time_window(df: pd.DataFrame, y: pd.Series, window_size: int):
    X_list, y_list = [], []
    feature_cols = df.columns.tolist()

    for i in range(len(df) - window_size):
        window = df.iloc[i: i + window_size][feature_cols].copy()
        window['id'] = i
        X_list.append(window)
        y_list.append(y.iloc[i + window_size])

    X = pd.concat(X_list, ignore_index=True)
    y_out = pd.Series(y_list, index=range(len(y_list)))
    return X, y_out


def extract_tsfresh(df_train, df_test, y_train, y_test, window_size: int = WINDOW_SIZE):
    print("[baseline/tsfresh] Tạo time windows...")
    X_tr_win, y_tr_win = _make_time_window(df_train, y_train, window_size)
    X_te_win, y_te_win = _make_time_window(df_test,  y_test,  window_size)

    print("[baseline/tsfresh] Đang extract features (có thể mất vài phút)...")
    X_tr_feat = extract_features(X_tr_win, column_id='id')
    X_te_feat = extract_features(X_te_win, column_id='id')

    X_tr_feat = impute(X_tr_feat)
    X_te_feat = impute(X_te_feat)

    print("[baseline/tsfresh] Đang select features...")
    X_tr_sel = select_features(X_tr_feat, y_tr_win)
    X_te_sel = X_te_feat[X_tr_sel.columns]

    X_tr_sel = _clean_names(X_tr_sel)
    X_te_sel = _clean_names(X_te_sel)

    X_tr_res, y_tr_res = apply_smote(X_tr_sel, y_tr_win)
    return X_tr_res, X_te_sel, y_tr_res, y_te_win


# ==============================================================================
# TSFEL
# ==============================================================================
def extract_tsfel(df_train, df_test, y_train, y_test, window_size: int = WINDOW_SIZE):
    """
    Trả về (X_train_res, X_test_clean, y_train_res, y_test_win)
    đã qua: windowing → tsfel extract → impute → SMOTE.
    """
    def _extract(df, y):
        X_windows, y_list = [], []
        for i in range(len(df) - window_size):
            X_windows.append(df.iloc[i: i + window_size].copy())
            y_list.append(y.iloc[i + window_size])
        cfg = tsfel.get_features_by_domain()
        cfg = {'statistical': cfg['statistical'], 'temporal': cfg['temporal']}
        X_feat = tsfel.time_series_features_extractor(cfg, X_windows, fs=1)
        return X_feat, np.array(y_list)

    print("[baseline/tsfel] Đang extract features train...")
    X_tr_feat, y_tr_win = _extract(df_train, y_train)
    print("[baseline/tsfel] Đang extract features test...")
    X_te_feat, y_te_win = _extract(df_test, y_test)


    imputer = SimpleImputer(strategy='median')
    X_tr_feat = pd.DataFrame(imputer.fit_transform(X_tr_feat), columns=X_tr_feat.columns)
    X_te_feat = pd.DataFrame(imputer.transform(X_te_feat),     columns=X_te_feat.columns)

    X_tr_feat = _clean_names(X_tr_feat)
    X_te_feat = _clean_names(X_te_feat)

    X_tr_res, y_tr_res = apply_smote(X_tr_feat, y_tr_win)
    return X_tr_res, X_te_feat, y_tr_res, y_te_win


# ==============================================================================
# HELPER
# ==============================================================================
def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df


# ==============================================================================
# PIPELINE CHÍNH
# ==============================================================================
def run_baseline(data_dir: str = None, extractor: str = 'both') -> dict:
    data_dir = data_dir or DATA_DIR

    df_train, df_test, y_train, y_test = load_data(data_dir)
    results = {}

    if extractor in ('tsfresh', 'both'):
        print("\n=== BASELINE – TSFRESH ===")
        X_tr, X_te, y_tr, y_te = extract_tsfresh(df_train, df_test, y_train, y_test)
        res = evaluate(X_tr, X_te, y_tr, y_te)
        res.insert(0, 'Feature', 'TSFRESH')
        results['tsfresh'] = res
        print(res.to_string(index=False))

    if extractor in ('tsfel', 'both'):
        print("\n=== BASELINE – TSFEL ===")
        X_tr, X_te, y_tr, y_te = extract_tsfel(df_train, df_test, y_train, y_test)
        res = evaluate(X_tr, X_te, y_tr, y_te)
        res.insert(0, 'Feature', 'TSFEL')
        results['tsfel'] = res
        print(res.to_string(index=False))

    return results
