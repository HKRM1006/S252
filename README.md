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
Chạy notebook [S252]_DADN_EDA_&_Preprocessing.ipynb để tải dữ liệu, EDA và tiền xử lý chúng.

Chạy file run.py để thực hiện feature extraction và đánh giá
```bash
python run.py --mode [baseline|custom|all] [options]
```
### 📋 Các tham số chính

| Tham số | Lựa chọn | Mặc định | Mô tả |
|:--- |:--- |:--- |:--- |
| `--mode` | `baseline`, `custom`, `all` | **Bắt buộc** | Chọn pipeline để thực thi. |
| `--extractor` | `tsfresh`, `tsfel`, `both` | `both` | Công cụ trích xuất đặc trưng (chỉ dành cho Baseline). |
| `--data_dir` | (Đường dẫn) | `None` | Thư mục chứa dữ liệu đầu vào (df_train, y_train...). Chỉ dùng khi đặt dữ liệu ở thư mục khác.|
| `--output` | (Tên file .csv) | `results.csv` | Tên file lưu kết quả tổng hợp sau khi chạy. |
