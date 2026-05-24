import argparse
import os

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import pandas as pd
from src.features.custom import run_custom
from src.features.custom_v2 import run_custom_v2
from src.features.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(
        description='Rain Prediction – chọn pipeline để chạy'
    )
    parser.add_argument(
        '--mode',
        choices=['baseline', 'custom', 'custom_v2', 'compare', 'all'],
        required=True,
        help='Pipeline cần chạy: baseline | custom | custom_v2 | compare | all',
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
    parser.add_argument(
        '--num_features_v2',
        type=int,
        default=60,
        help='Số feature chọn cho custom_v2 (default: 60)',
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
    if args.mode in ('custom', 'compare', 'all'):
        df_custom = run_custom(data_dir=args.data_dir)
        all_results.append(df_custom)

    if args.mode in ('custom_v2', 'compare', 'all'):
        df_custom_v2 = run_custom_v2(
            data_dir=args.data_dir,
            num_features=args.num_features_v2,
        )
        all_results.append(df_custom_v2)

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
