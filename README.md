# Rain Prediction – Modular Python Project

## Cấu trúc thư mục

```
DADN-252/
├── data/
│   ├── raw/                         
│   └── processed/                   # CSV được tạo bởi preprocessing
├── notebook/                        # Jupyter notebooks
├── src/
│   ├── data/
│   │   └── preprocessing.py         # KDD pipeline: load → cleanse → reduce → normalize → split → lưu CSV
│   ├── eval/
│   │   └── eval.py                  # MODELS + hàm evaluate() dùng chung
│   ├── features/
│   │   ├── baseline.py              # TSFRESH / TSFEL extraction
│   │   └── custom.py                # Feature engineering thủ công
│   └── util.py                      # load_data / normalize_features / apply_smote
├── .gitignore
├── README.md
├── requirements.txt
└── run.py                           # Program
```

## Cài đặt
```bash
pip install tsfresh tsfel xgboost lightgbm imbalanced-learn scikit-learn pandas numpy
```
## Cách sử dụng
Tải dữ liệu từ Kaggle về thư mục data/raw/
```bash
%run src/data/download_data.py
```

Tiền xử lý dữ liệu và lưu vào data/processed/. Trong ví dụ, nhóm đang lọc trạm A101 và giữ 20% dữ liệu có thời gian gần nhất.
```bash
%run src/data/preprocessing.py --station A101 --keep 0.2
```

Chạy notebook [S252]_DADN_EDA_&_Preprocessing.ipynb để thực hiện EDA.

Chạy file run.py để thực hiện feature extraction và đánh giá
```bash
python run.py --mode [baseline|custom|custom_v2|compare|all] [options]
```
### 📋 Các tham số chính

| Tham số | Lựa chọn | Mặc định | Mô tả |
|:--- |:--- |:--- |:--- |
| `--mode` | `baseline`, `custom`, `custom_v2`, `compare`, `all` | **Bắt buộc** | Chọn pipeline để thực thi. `compare` chạy Custom hiện có và Custom_V2 mới để so sánh trực tiếp. |
| `--extractor` | `tsfresh`, `tsfel`, `both` | `both` | Công cụ trích xuất đặc trưng (chỉ dành cho Baseline). |
| `--data_dir` | (Đường dẫn) | `None` | Thư mục chứa dữ liệu đầu vào (df_train, y_train...). Chỉ dùng khi đặt dữ liệu ở thư mục khác.|
| `--output` | (Tên file .csv) | `results.csv` | Tên file lưu kết quả tổng hợp sau khi chạy. |
| `--num_features_v2` | (Số nguyên) | `60` | Số đặc trưng chọn bằng mutual information cho bộ Custom_V2. |

## Custom_V2

`src/features/custom_v2.py` bổ sung một hướng trích xuất đặc trưng mới để so sánh với `Custom` hiện có:

- Feature set: lag 1/2/3/6/12/24/48h, rolling statistics 3/6/12/24/48h, precipitation memory, wet/dry streak, pressure tendency, humidity/temperature/dew-point/wind interaction, cyclical time features.
- Model mới trong `src/eval/eval.py`: `ET` - ExtraTreesClassifier với `class_weight='balanced'`.
- Metrics bổ sung: Balanced Accuracy, MCC, ROC-AUC, PR-AUC bên cạnh Accuracy, Precision, Recall, F1.

Chạy so sánh:
```bash
python run.py --mode compare --output compare_results.csv
```
