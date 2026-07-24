# 步骤6：模型选型与基线训练产出物

## 产出物1：各模型基线版本

| model_family | model | baseline_version_path | train_data | validation_data | validation_rows |
| --- | --- | --- | --- | --- | --- |
| traditional_ml | lightgbm | output/baseline_models/lightgbm_baseline.joblib | output/train_smote_r10.parquet | output/val.parquet | 936641 |
| traditional_ml | xgboost | output/baseline_models/xgboost_baseline.joblib | output/train_smote_r10.parquet | output/val.parquet | 936641 |
| traditional_ml | logistic_regression | output/baseline_models/logistic_regression_baseline.joblib | output/train_smote_r10.parquet | output/val.parquet | 936641 |
| deep_sequence | din | output/sequence_models_din_trial/din_baseline.pt | output/train.parquet + output/cleaned_data.parquet history | output/val.parquet first 50000 rows + history | 50000 |
| deep_sequence | lstm | output/sequence_models_lstm_trial/lstm_baseline.pt | output/train.parquet + output/cleaned_data.parquet history | output/val.parquet first 50000 rows + history | 50000 |
| deep_sequence | gru | output/sequence_models_gru_trial/gru_baseline.pt | output/train.parquet + output/cleaned_data.parquet history | output/val.parquet first 50000 rows + history | 50000 |

## 产出物2：基线性能对比表（默认阈值 0.5）

| model_family | model | validation_rows | roc_auc | pr_auc_ap | log_loss | precision_at_0_5 | recall_at_0_5 | f1_at_0_5 | baseline_version_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traditional_ml | lightgbm | 936641 | 0.992579 | 0.124940 | 0.011659 | 0.129056 | 0.472711 | 0.202756 | output/baseline_models/lightgbm_baseline.joblib |
| traditional_ml | xgboost | 936641 | 0.992698 | 0.124751 | 0.014061 | 0.104863 | 0.571303 | 0.177201 | output/baseline_models/xgboost_baseline.joblib |
| traditional_ml | logistic_regression | 936641 | 0.979114 | 0.070285 | 0.055673 | 0.053911 | 0.761444 | 0.100693 | output/baseline_models/logistic_regression_baseline.joblib |
| deep_sequence | din | 50000 | 0.742454 | 0.028879 | 0.451894 | 0.003166 | 0.500000 | 0.006292 | output/sequence_models_din_trial/din_baseline.pt |
| deep_sequence | lstm | 50000 | 0.607214 | 0.026593 | 0.172282 | 0.004408 | 0.304348 | 0.008690 | output/sequence_models_lstm_trial/lstm_baseline.pt |
| deep_sequence | gru | 50000 | 0.639439 | 0.016474 | 0.110634 | 0.006632 | 0.304348 | 0.012981 | output/sequence_models_gru_trial/gru_baseline.pt |

## 产出物3：各模型最优基线阈值（按 F1 选择）

| model_family | model | validation_rows | threshold | precision | recall | f1 | accuracy | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traditional_ml | lightgbm | 936641 | 0.600000 | 0.148401 | 0.363556 | 0.210768 | 0.996698 | 933135 | 2370 | 723 | 413 |
| traditional_ml | xgboost | 936641 | 0.700000 | 0.145100 | 0.333627 | 0.202241 | 0.996808 | 933272 | 2233 | 757 | 379 |
| traditional_ml | logistic_regression | 936641 | 0.900000 | 0.080799 | 0.491197 | 0.138771 | 0.992605 | 929157 | 6348 | 578 | 558 |
| deep_sequence | gru | 50000 | 0.900000 | 0.013917 | 0.152174 | 0.025501 | 0.989300 | 49458 | 496 | 39 | 7 |
| deep_sequence | lstm | 50000 | 0.900000 | 0.012896 | 0.239130 | 0.024472 | 0.982460 | 49112 | 842 | 35 | 11 |
| deep_sequence | din | 50000 | 0.900000 | 0.006944 | 0.326087 | 0.013599 | 0.956480 | 47809 | 2145 | 31 | 15 |

## 产出物4：阈值调整前后性能对比

下表对比默认阈值 `0.5` 与按验证集 F1 选择的阈值结果。该步骤只改变概率判别阈值，不重新训练模型，也不改变模型参数。

| model_family | model | validation_rows | default_threshold | default_precision | default_recall | default_f1 | selected_threshold | selected_precision | selected_recall | selected_f1 | f1_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traditional_ml | lightgbm | 936641 | 0.500000 | 0.129056 | 0.472711 | 0.202756 | 0.600000 | 0.148401 | 0.363556 | 0.210768 | 0.008012 |
| traditional_ml | xgboost | 936641 | 0.500000 | 0.104863 | 0.571303 | 0.177201 | 0.700000 | 0.145100 | 0.333627 | 0.202241 | 0.025040 |
| traditional_ml | logistic_regression | 936641 | 0.500000 | 0.053911 | 0.761444 | 0.100693 | 0.900000 | 0.080799 | 0.491197 | 0.138771 | 0.038079 |
| deep_sequence | gru | 50000 | 0.500000 | 0.006632 | 0.304348 | 0.012981 | 0.900000 | 0.013917 | 0.152174 | 0.025501 | 0.012520 |
| deep_sequence | lstm | 50000 | 0.500000 | 0.004408 | 0.304348 | 0.008690 | 0.900000 | 0.012896 | 0.239130 | 0.024472 | 0.015781 |
| deep_sequence | din | 50000 | 0.500000 | 0.003166 | 0.500000 | 0.006292 | 0.900000 | 0.006944 | 0.326087 | 0.013599 | 0.007307 |

## 阈值候选集合

`0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90`

完整多阈值结果见 `output/step6_baseline_deliverables/all_baseline_threshold_metrics.csv`。

## 对比口径说明

- 传统机器学习模型使用 `train_smote_r10.parquet` 训练，在完整 `val.parquet` 上验证。
- 深度学习序列模型使用 `train.parquet` 的样本索引，并从 `cleaned_data.parquet` 构造真实用户历史序列；本轮 trial 在 200000 条训练样本和 50000 条验证样本上完成。
- 默认阈值 0.5 用于统一基线对比；最优基线阈值表用于观察不同模型在验证集上的 Precision/Recall/F1 权衡。
- 因两类模型的数据组织方式不同，报告中建议分别比较传统模型内部、序列模型内部；跨体系结果用于方向判断，不作为最终优劣结论。
- 当前所有模型均为基线版本，未做系统性超参数调优。
