import argparse
import os
import pandas as pd
from src.features.custom import run_custom
from src.features.baseline import run_baseline
def main():
    parser = argparse.ArgumentParser(
        description='Rain Prediction – chọn pipeline để chạy'
    )
    parser.add_argument(
        '--mode',
        choices=['baseline', 'custom', 'all'],
        required=True,
        help='Pipeline cần chạy: baseline | custom | all',
    )
    parser.add_argument(
        '--extractor',
        choices=['tsfresh', 'tsfel', 'both'],
        default='both',
        help='(Chỉ dùng khi mode=baseline) Extractor: tsfresh | tsfel | both',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Thư mục chứa df_train.csv / df_test.csv / y_train.csv / y_test.csv',
    )
    parser.add_argument(
        '--output',
        default='results.csv',
        help='File CSV lưu kết quả tổng hợp (default: results.csv)',
    )
    args = parser.parse_args()

    all_results = []

    # ── BASELINE ────────────────────────────────────────────────────────────
    if args.mode in ('baseline', 'all'):
        results_dict = run_baseline(
            data_dir  = args.data_dir,
            extractor = args.extractor if args.mode == 'baseline' else 'both',
        )
        for df in results_dict.values():
            all_results.append(df)

    # ── CUSTOM ──────────────────────────────────────────────────────────────
    if args.mode in ('custom', 'all'):
        df_custom = run_custom(data_dir=args.data_dir)
        all_results.append(df_custom)

    # ── TỔNG HỢP & LƯU ─────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\n{'='*60}")
        print("KẾT QUẢ TỔNG HỢP")
        print('='*60)
        print(combined.to_string(index=False))
        print(f"\nĐã lưu kết quả vào '{args.output}'")


if __name__ == '__main__':
    main()
