import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# ==============================================================================
# LOAD DỮ LIỆU ĐÃ PREPROCESSING
# ==============================================================================
def load_data(data_dir: str = DATA_DIR):
    """
    Đọc 4 file CSV đã được tạo bởi src/data/preprocessing.py.
    """
    df_train = pd.read_csv(os.path.join(data_dir, 'df_train.csv'), index_col=0, parse_dates=True)
    df_test  = pd.read_csv(os.path.join(data_dir, 'df_test.csv'),  index_col=0, parse_dates=True)
    y_train  = pd.read_csv(os.path.join(data_dir, 'y_train.csv'),  index_col=0, parse_dates=True).squeeze()
    y_test   = pd.read_csv(os.path.join(data_dir, 'y_test.csv'),   index_col=0, parse_dates=True).squeeze()
    return df_train, df_test, y_train, y_test


# ==============================================================================
# NORMALIZE (fit trên train, transform test)  — dùng sau feature extraction
# ==============================================================================
def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Quy trình chuẩn hóa chung
    """
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


# ==============================================================================
# SMOTE — cân bằng class trên tập train
# ==============================================================================
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Áp dụng SMOTE lên (X_train, y_train).
    Trả về (X_resampled, y_resampled).
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
