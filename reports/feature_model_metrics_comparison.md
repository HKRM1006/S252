# Custom vs Custom_V2 Comparison

Dataset setup:
- Station: A101
- Data slice: latest 20% of station data
- Train/test split: chronological 80/20
- Train rows after preprocessing: 29,420
- Test rows after preprocessing: 7,356

Implemented work:
- New feature set: `Custom_V2` in `src/features/custom_v2.py`
- New model: `ET` (`ExtraTreesClassifier`) in `src/eval/eval.py`
- New metrics: Balanced Accuracy, MCC, ROC-AUC, PR-AUC
- Comparison command: `python run.py --mode compare --output compare_results.csv`

Best result by macro F1:

| Feature | Model | Accuracy | Balanced Accuracy | F1-Score | MCC | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom | LGBM | 0.9370 | 0.7487 | 0.7962 | 0.6131 | 0.8772 | 0.6782 |
| Custom_V2 | LGBM | 0.9320 | 0.7684 | 0.7968 | 0.5997 | 0.8774 | 0.6471 |

Best result by balanced accuracy / macro recall:

| Feature | Model | Accuracy | Balanced Accuracy | F1-Score | MCC | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom_V2 | LR | 0.8195 | 0.8034 | 0.6821 | 0.4333 | 0.8771 | 0.6162 |

Conclusion:
- If the main score is macro F1, `Custom_V2 + LGBM` is the best result: 0.7968 vs 0.7962 for `Custom + LGBM`.
- If detecting rainy cases matters more, `Custom_V2` is clearly better on Balanced Accuracy / macro Recall: 0.7684 with LGBM and 0.8034 with LR vs 0.7487 for the best old Custom model.
- The tradeoff is lower PR-AUC, MCC, and raw accuracy compared with `Custom + LGBM`, so `Custom + LGBM` remains preferable when precision and ranking quality are prioritized.
