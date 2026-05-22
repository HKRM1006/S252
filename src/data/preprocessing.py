import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.util import DATA_DIR, RAW_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

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

COLUMNS_TO_DROP = [
    'Temperature_Max', 'Temperature_Min',
    'Dew_Point_Max',   'Dew_Point_Min',
    'Humidity_Max',    'Humidity_Min',
    'Pressure_Max',    'Pressure_Min',
    'region', 'state', 'station', 'station_code',
    'latitude', 'longitude', 'height',
]

# ==============================================================================
# BƯỚC 1 – LOAD DỮ LIỆU THÔ
# ==============================================================================
def load_station(file_path: str, station_code: str = 'A101', percent_to_keep: float = 0.2) -> pd.DataFrame:
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

    # --- ĐIỂM SỬA ---
    # Sắp xếp index (thời gian) từ cũ đến mới để đảm bảo tail() luôn lấy dữ liệu gần nhất
    df_station.sort_index(inplace=True)

    # Giữ percent_to_keep dòng cuối (Thời gian gần nhất)
    keep_size = int(len(df_station) * percent_to_keep)
    df_station = df_station.tail(keep_size).copy()

    print(f"[load_station] Trạm {station_code}: {df_station.shape} "
          f"({df_station.index.min()} → {df_station.index.max()})")
    return df_station

# ==============================================================================
# BƯỚC 2 – HANDLING MISSING VALUES
# ==============================================================================
def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.asfreq('1h') 

    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')
    df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
    df_clean.dropna(subset=numeric_cols, inplace=True)
    
    # Logic tạo biến Rain cho bài toán dự báo t+1
    # df_clean['Rain'] = (df_clean['Precipitation'].shift(-1) > 0).astype(int)
    # df_clean = df_clean.iloc[:-1].copy()
    df_clean['Rain'] = (df_clean['Precipitation'] > 0).astype(int)
    df_clean = df_clean.iloc[:-1].copy()

    print(f"[cleanse] Sau khi xử lý dữ liệu thiếu: {df_clean.shape}")
    return df_clean

# ==============================================================================
# BƯỚC 3 – REDUCTION
# ==============================================================================
def reduce(df: pd.DataFrame) -> pd.DataFrame:
    df_reduced = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"[reduce] Sau reduction: {df_reduced.shape}")
    return df_reduced

# ==============================================================================
# BƯỚC 4 – NORMALIZATION
# ==============================================================================
def normalize(df_train: pd.DataFrame, df_test: pd.DataFrame):
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
            f"\n[LỖI ĐỒNG BỘ DATA] Không tìm thấy file dữ liệu gốc tại: '{raw_path}'\n"
            f"Vui lòng chạy file download_data.py trước hoặc kiểm tra lại tham số '--raw_path'."
        )
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Chạy lần lượt qua các bước
    df = load_station(raw_path, station_code=station_code, percent_to_keep=percent_to_keep)
    df = interpolate_data(df)
    df = reduce(df)
    
    # Chia tập & Chuẩn hóa
    y = df.pop('Rain')
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)
    y_train,  y_test  = train_test_split(y,  test_size=test_size, shuffle=False)
    
    df_train, df_test = normalize(df_train, df_test)
    
    # Lưu file
    df_train.to_csv(os.path.join(out_dir, 'df_train.csv'))
    df_test.to_csv(os.path.join(out_dir,  'df_test.csv'))
    y_train.to_csv(os.path.join(out_dir,  'y_train.csv'))
    y_test.to_csv(os.path.join(out_dir,   'y_test.csv'))

    print(f"\n[preprocessing] Đã lưu 4 file vào '{out_dir}'")
    print(f"  df_train : {df_train.shape}  ({df_train.index.min()} → {df_train.index.max()})")
    print(f"  df_test  : {df_test.shape}   ({df_test.index.min()}  → {df_test.index.max()})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing pipeline – lưu CSV vào data/processed/')
    
    default_raw_path = os.path.join(RAW_DIR, 'north.csv')
    parser.add_argument('--raw_path', default=default_raw_path, help=f'Đường dẫn tới file CSV gốc (default: {default_raw_path})')
    
    parser.add_argument('--station',  default='A101',  help='Mã trạm (default: A101)')
    parser.add_argument('--keep',     type=float, default=0.2, help='Tỉ lệ dữ liệu giữ lại (default: 0.2)')
    parser.add_argument('--test_size',type=float, default=0.2, help='Tỉ lệ test split (default: 0.2)')
    parser.add_argument('--out_dir',  default=DATA_DIR, help=f'Thư mục lưu output (default: {DATA_DIR})')
    args = parser.parse_args()
    
    run_preprocessing(
        raw_path       = args.raw_path,
        station_code   = args.station,
        percent_to_keep= args.keep,
        test_size      = args.test_size,
        out_dir        = args.out_dir,
    )