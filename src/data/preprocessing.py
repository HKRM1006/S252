import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.util import DATA_DIR, RAW_DIR

# ==============================================================================
# CONSTANTS
# ==============================================================================
RENAME_DICT = {
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)':                     'Precipitation',
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)':'Pressure',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)':      'Pressure_Max',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)':     'Pressure_Min',
    'RADIACAO GLOBAL (Kj/m²)':                              'Solar_Radiation',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)':         'Temperature',
    'TEMPERATURA DO PONTO DE ORVALHO (°C)':                 'Dew_Point',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)':           'Temperature_Max',
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)':           'Temperature_Min',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)':     'Dew_Point_Max',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)':     'Dew_Point_Min',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)':             'Humidity_Max',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)':             'Humidity_Min',
    'UMIDADE RELATIVA DO AR, HORARIA (%)':                  'Humidity',
    'VENTO, DIREÇÃO HORARIA (gr) (° (gr))':                 'Wind_Direction',
    'VENTO, RAJADA MAXIMA (m/s)':                           'Wind_Gust',
    'VENTO, VELOCIDADE HORARIA (m/s)':                      'Wind_Speed',
}

# Cột loại bỏ: dư thừa (r>0.9) + metadata + tránh data leakage
COLUMNS_TO_DROP = [
    'Temperature_Max', 'Temperature_Min',
    'Dew_Point_Max',   'Dew_Point_Min',
    'Humidity_Max',    'Humidity_Min',
    'Pressure_Max',    'Pressure_Min',
    'Solar_Radiation',                      
    'region', 'state', 'station', 'station_code',
    'latitude', 'longitude', 'height',
]

def load_raw(file_path: str, station_code: str = 'A101', percent_to_keep: float = 0.2) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Thay mã lỗi INMET bằng NaN
    df.replace(-9999.0, np.nan, inplace=True)
    df.replace(-9999,   np.nan, inplace=True)

    df.rename(columns=RENAME_DICT, inplace=True)

    # Gộp Date + Time → DatetimeIndex
    df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Data', 'Hora', 'index'], inplace=True, errors='ignore')

    # Lọc trạm
    df_station = df[df['station_code'] == station_code].copy()

    # Giữ percent_to_keep dòng cuối
    keep_size = int(len(df_station) * percent_to_keep)
    df_station = df_station.tail(keep_size).copy()

    print(f"[load_raw] Trạm {station_code}: {df_station.shape} "
          f"({df_station.index.min()} → {df_station.index.max()})")
    return df_station


# ==============================================================================
# BƯỚC 2 – CLEANSING
# ==============================================================================
def cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.asfreq('1h')

    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
    df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
    df_clean.dropna(subset=numeric_cols, inplace=True)
    df_clean['Rain'] = (df_clean['Precipitation'].shift(-1) > 0).astype(int)
    df_clean = df_clean.iloc[:-1].copy()

    print(f"[cleanse] Sau cleansing: {df_clean.shape}")
    return df_clean


def reduce(df: pd.DataFrame) -> pd.DataFrame:
    df_reduced = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"[reduce] Sau reduction: {df_reduced.shape}")
    return df_reduced


def normalize(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Fit StandardScaler trên train, transform cả train + test.
    Không scale cột Rain (target) và Precipitation.
    Trả về (df_train_scaled, df_test_scaled).
    """
    exclude = {'Rain', 'Precipitation'}
    numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features_to_scale = [c for c in numeric_cols if c not in exclude]

    scaler = StandardScaler()
    df_train = df_train.copy()
    df_test  = df_test.copy()

    df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])
    df_test[features_to_scale]  = scaler.transform(df_test[features_to_scale])

    print(f"[normalize] Đã scale {len(features_to_scale)} cột.")
    return df_train, df_test


# ==============================================================================
# PIPELINE CHÍNH
# ==============================================================================
def run_preprocessing(
    raw_path: str = RAW_DIR + "/north.csv",
    station_code: str  = 'A101',
    percent_to_keep: float = 0.2,
    test_size: float   = 0.2,
    out_dir: str       = DATA_DIR,
):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"\n [LỖI ĐỒNG BỘ DATA] Không tìm thấy file dữ liệu gốc tại: '{raw_path}'\n"
            f" Nguyên nhân: Có thể bạn nhập sai đường dẫn ở lệnh --raw_path, hoặc chưa download dữ liệu về trước đó.\n"
            f"Cách sửa: Vui lòng chạy 'python download_data.py' để tải dữ liệu về đúng kho, "
            f"hoặc kiểm tra lại tham số truyền vào!"
        )
    os.makedirs(out_dir, exist_ok=True)
    df = load_raw(raw_path, station_code=station_code, percent_to_keep=percent_to_keep)
    df = cleanse(df)
    df = reduce(df)
    y = df.pop('Rain')
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)
    y_train,  y_test  = train_test_split(y,  test_size=test_size, shuffle=False)
    df_train, df_test = normalize(df_train, df_test)
    df_train.to_csv(os.path.join(out_dir, 'df_train.csv'))
    df_test.to_csv(os.path.join(out_dir,  'df_test.csv'))
    y_train.to_csv(os.path.join(out_dir,  'y_train.csv'))
    y_test.to_csv(os.path.join(out_dir,   'y_test.csv'))

    print(f"\n[preprocessing] Đã lưu 4 file vào '{out_dir}'")
    print(f"  df_train : {df_train.shape}  ({df_train.index.min()} → {df_train.index.max()})")
    print(f"  df_test  : {df_test.shape}   ({df_test.index.min()}  → {df_test.index.max()})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing pipeline – lưu CSV vào data/')
    parser.add_argument('--raw_path', default=RAW_DIR + "/north.csv", help='Đường dẫn tới file CSV gốc (north.csv)')
    parser.add_argument('--station',  default='A101',  help='Mã trạm (default: A101)')
    parser.add_argument('--keep',     type=float, default=0.2, help='Tỉ lệ dữ liệu giữ lại (default: 0.2)')
    parser.add_argument('--test_size',type=float, default=0.2, help='Tỉ lệ test split (default: 0.2)')
    parser.add_argument('--out_dir',  default=DATA_DIR, help='Thư mục lưu output')
    args = parser.parse_args()
    run_preprocessing(
        raw_path       = args.raw_path,
        station_code   = args.station,
        percent_to_keep= args.keep,
        test_size      = args.test_size,
        out_dir        = args.out_dir,
    )
