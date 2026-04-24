"""
cần 1 hàm run_custom chạy và trả về DataFrame chứa kết quả đánh giá quy trình
Sử dụng hàm evalute trong src/eval/eval.py để đánh giá
Các hàm load dữ liệu, Normalize, SMOTE nằm trong utils
"""
from src.util import DATA_DIR
from src.util import load_data, apply_smote
from src.eval.eval import evaluate
def run_custom(data_dir: str = None) -> dict:
    data_dir = data_dir or DATA_DIR
    df_train, df_test, y_train, y_test = load_data(data_dir)
    ### Thuc hien extract feature

    ###
    results = evaluate(X_train_res, X_test_norm, y_train_res, y_test)
    results.insert(0, 'Feature', 'Custom')
    print(results.to_string(index=False))
 
    return results