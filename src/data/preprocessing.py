import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def clean_data(df):
    """PHẦN 1: TIỀN XỬ LÝ TOÀN BỘ 27 CỘT"""
    print("--- BẮT ĐẦU QUY TRÌNH EDA TOÀN DIỆN CHO 27 CỘT ---")

    # Chuyển đổi mã lỗi -9999 thành NaN
    df.replace(-9999.0, np.nan, inplace=True)
    df.replace(-9999, np.nan, inplace=True)

    # Đổi tên toàn bộ 27 cột sang tiếng Anh
    full_rename_dict = {
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Precipitation',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'Pressure',
        'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'Pressure_Max',
        'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'Pressure_Min',
        'RADIACAO GLOBAL (Kj/m²)': 'Solar_Radiation',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'Temperature',
        'TEMPERATURA DO PONTO DE ORVALHO (°C)': 'Dew_Point',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)': 'Temperature_Max',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)': 'Temperature_Min',
        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)': 'Dew_Point_Max',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)': 'Dew_Point_Min',
        'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'Humidity_Max',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)': 'Humidity_Min',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'Humidity',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'Wind_Direction',
        'VENTO, RAJADA MAXIMA (m/s)': 'Wind_Gust',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'Wind_Speed'
    }
    df.rename(columns=full_rename_dict, inplace=True)

    # Gom Date và Time thành Datetime Index
    df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Data', 'Hora', 'index'], inplace=True, errors='ignore')

    # Phân nhóm cột & Lọc 1 Trạm (A101) để phân tích chuỗi thời gian
    df_station = df[df['station_code'] == 'A101'].copy()

    # GIẢM KÍCH THƯỚC DỮ LIỆU (Tìm đoạn 20% liên tục có ít Missing Value nhất)
    percent_to_keep = 0.2
    keep_size = int(len(df_station) * percent_to_keep)

    # Lấy keep_size dòng cuối cùng
    df_station = df_station.tail(keep_size).copy()

    print(f"=> Kích thước dữ liệu trạm A101 sau khi cắt: {df_station.shape} dòng.")
    print("\n--- SẴN SÀNG CHO EDA VÀ PREPROCESSING ---")
    
    return df_station


def interpolate_data(df_station):
    """DATA CLEANSING (Làm sạch & Nội suy Missing Data)"""
    print("--- BẮT ĐẦU QUY TRÌNH DATA PREPROCESSING THEO CHUẨN KDD ---")
    
    df_clean = df_station.copy()
    df_clean = df_clean.asfreq('1h')
    print("Đã rà soát timeline và tạo các dòng trống (NaN) cho các mốc thời gian bị khuyết.")

    # Chọn các biến số học cần nội suy (loại bỏ biến dạng chuỗi/metadata)
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Nội suy tuyến tính theo thời gian (Time-series interpolation) cho dữ liệu bị thiếu
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='time')

    # Điền các giá trị NaN ở biên (nếu có) bằng Forward Fill / Backward Fill
    df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
    df_clean.dropna(subset=numeric_cols, inplace=True)

    print("1. Đã hoàn thành Data Cleansing: Xử lý dữ liệu thiếu bằng Nội suy.")
    return df_clean


def split_and_save(df_reduced, y_rain):
    """Chia tập train/test và lưu file vào cùng thư mục với script này."""
    print("\n--- BẮT ĐẦU CHIA TẬP VÀ LƯU FILE ---")
    
    df_train, df_test = train_test_split(df_reduced, test_size=0.2, shuffle=False)
    y_train, y_test = train_test_split(y_rain, test_size=0.2, shuffle=False)

    print(f" Train X: {df_train.shape} | Test X: {df_test.shape}")
    
    # Lấy đường dẫn thư mục chứa file preprocessing.py hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tạo đường dẫn lưu file
    paths = {
        'df_train': os.path.join(current_dir, 'df_train_preprocessed.csv'),
        'df_test': os.path.join(current_dir, 'df_test_preprocessed.csv'),
        'y_train': os.path.join(current_dir, 'y_train_preprocessed.csv'),
        'y_test': os.path.join(current_dir, 'y_test_preprocessed.csv'),
    }

    df_train.to_csv(paths['df_train'])
    df_test.to_csv(paths['df_test'])
    y_train.to_csv(paths['y_train'])
    y_test.to_csv(paths['y_test'])

    print(f"\n=> Đã lưu 4 files thành công tại: {current_dir}")
    return df_train, df_test, y_train, y_test