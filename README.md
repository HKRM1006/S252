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
│   └── util.py                      # Các bước giống nhau giữa 2 pipeline: load_data / normalize_features / apply_smote
├── .gitignore
├── README.md
├── requirements.txt
└── run.py                           # Program
```

## Cài đặt
```bash
pip install tsfresh tsfel xgboost lightgbm imbalanced-learn scikit-learn pandas numpy
```
