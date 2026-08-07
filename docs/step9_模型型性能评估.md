# Step9 模型效果综合评估报告

## 1. 评估目标

本步骤使用独立测试集评估最终模型的泛化能力。评估过程不重新训练模型，不使用测试集选择模型参数；测试集只用于最终性能报告。

## 2. 评估配置

- 测试集路径：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/test.parquet`
- 模型路径：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib`
- 预测文件路径：`None`
- 输出目录：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/step9_test_model_evaluation`
- 测试集样本数：468,316
- 正样本数：449
- 负样本数：467,867
- 正样本占比：0.095875%
- 特征数：24
- 标签列：`label`
- 固定评估阈值：0.2
- 阈值来源：`internal_tuning_validation`

## 3. 核心评估指标

| roc_auc | pr_auc_ap | log_loss | threshold | accuracy | precision | recall | f1 | tn | fp | fn | tp | model_path | test_path | prediction_source | target_col | threshold_source | n_rows | positive_count | negative_count | positive_rate | feature_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.994194 | 0.131205 | 0.005174 | 0.200000 | 0.995866 | 0.118130 | 0.512249 | 0.191987 | 466150 | 1717 | 219 | 230 | /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | /Users/yangzhuoyao/Desktop/阿里/项目1/output/test.parquet | model_predict_proba | label | internal_tuning_validation | 468316 | 449 | 467867 | 0.000959 | 24 |

说明：ROC-AUC 衡量模型整体排序能力，Accuracy、Precision、Recall 和 F1 衡量固定阈值下的分类效果。由于购买预测正负样本极不平衡，PR-AUC 也作为补充指标，用于观察模型对正样本的识别能力。

## 4. 阈值诊断

| model_path | threshold | accuracy | precision | recall | f1 | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.050000 | 0.989926 | 0.072673 | 0.808463 | 0.133358 | 463235 | 4632 | 86 | 363 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.100000 | 0.993225 | 0.094160 | 0.703786 | 0.166097 | 464827 | 3040 | 133 | 316 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.150000 | 0.994914 | 0.111379 | 0.616927 | 0.188692 | 465657 | 2210 | 172 | 277 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.200000 | 0.995866 | 0.118130 | 0.512249 | 0.191987 | 466150 | 1717 | 219 | 230 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.250000 | 0.996447 | 0.117202 | 0.414254 | 0.182711 | 466466 | 1401 | 263 | 186 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.300000 | 0.997055 | 0.131537 | 0.369710 | 0.194039 | 466771 | 1096 | 283 | 166 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.350000 | 0.997476 | 0.142439 | 0.325167 | 0.198100 | 466988 | 879 | 303 | 146 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.400000 | 0.997790 | 0.150358 | 0.280624 | 0.195804 | 467155 | 712 | 323 | 126 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.450000 | 0.998074 | 0.166421 | 0.251670 | 0.200355 | 467301 | 566 | 336 | 113 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.500000 | 0.998238 | 0.171329 | 0.218263 | 0.191969 | 467393 | 474 | 351 | 98 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.600000 | 0.998576 | 0.202186 | 0.164811 | 0.181595 | 467575 | 292 | 375 | 74 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.700000 | 0.998749 | 0.220408 | 0.120267 | 0.155620 | 467676 | 191 | 395 | 54 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.800000 | 0.998905 | 0.264706 | 0.080178 | 0.123077 | 467767 | 100 | 413 | 36 |
| /Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib | 0.900000 | 0.998969 | 0.276316 | 0.046771 | 0.080000 | 467812 | 55 | 428 | 21 |

在测试集上的 F1 诊断最优阈值为 0.45，对应 F1 为 0.200355。该结果只用于误差分析和阈值敏感性说明，不能反向用于重新选择模型参数，否则会引入测试集信息泄露。

## 5. 产出物

| 文件 | 说明 |
| --- | --- |
| `step9_generalization_metrics.csv` | 固定阈值下的综合评估指标 |
| `step9_threshold_metrics.csv` | 多阈值 Precision/Recall/F1 诊断结果 |
| `step9_test_predictions.csv` | 测试集逐样本预测概率与预测标签 |
| `step9_evaluation_setup.json` | 本次评估的数据、模型、阈值和耗时配置 |
| `step9_model_evaluation_report.md` | 本 Markdown 报告 |
