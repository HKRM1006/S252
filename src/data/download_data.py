import argparse
import os
import kagglehub
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.util import RAW_DIR

KAGGLE_DATASET = "PROPPG-PPG/hourly-weather-surface-brazil-southeast-region"
RAW_FILE_NAME = "north.csv"

def download_data(out_dir=RAW_DIR):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Đang tải dataset từ Kaggle ({KAGGLE_DATASET})...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    
    raw_path = os.path.join(path, RAW_FILE_NAME)
    dest_path = os.path.join(out_dir, RAW_FILE_NAME)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Không tìm thấy {RAW_FILE_NAME} trong gói dataset đã tải từ Kaggle!")
    shutil.copy(raw_path, dest_path)
    
    print(f"[+] Đã tải và lưu tại: {dest_path}")
    return dest_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script tải dữ liệu từ Kaggle về thư mục chỉ định')
    parser.add_argument('--out_dir', default=RAW_DIR, help=f'Thư mục lưu file raw tải về (default: {RAW_DIR})')
    args = parser.parse_args()
    download_data(out_dir=args.out_dir)