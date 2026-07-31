# Part7 调优模型评估补充

## 0. 实验口径

本部分对应“模型调优与融合方案”任务，主要比较传统机器学习模型、DIN 深度学习模型以及融合模型在购买预测任务中的表现。由于最终验证集 `output/val.parquet` 正样本占比仅约 0.1213%，本轮以 `PR-AUC / Average Precision` 作为主指标，同时报告 ROC-AUC、Log Loss 和 F1。

本轮代码统一采用以下防泄露口径：

| 数据 | 路径 | 样本量 | 正样本量 | 正样本占比 | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| 原始训练集 | `output/train.parquet` | 3,278,239 | 4,102 | 0.1251% | 内部抽样、调参、DIN 全量训练 |
| SMOTE 训练集 | `output/train_smote_r10.parquet` | 3,637,930 | 363,793 | 10.0000% | 传统机器学习最终训练 |
| 最终验证集 | `output/val.parquet` | 936,641 | 1,136 | 0.1213% | 只用于最终评估 |

传统机器学习和 DIN 调参均从 `train.parquet` 内部分层抽样 500,000 条，分层维度为：

```text
label × 日期 × 工作日/周末/特殊日 × 时段
```

每个时间层内部按 `last_time` 顺序切分为 80% 内部训练集和 20% 内部验证集。传统机器学习只对内部训练集做 SMOTE，内部验证集保持原始分布。`val.parquet` 不参与 Optuna 调参、阈值选择、DIN 早停或 Stacking 二层模型训练。

## 1. 传统机器学习调优

对应脚本：

```text
src/run_baseline_models.py
src/run_baseline_thresholds.py
src/run_optuna_tuned_models.py
```

### 1.1 数据与训练流程

传统机器学习部分先训练 LightGBM、XGBoost、Logistic Regression 三个 baseline，再使用 Optuna 对 LightGBM 和 XGBoost 进行调参。调参阶段只使用 `train.parquet` 内部 500,000 条分层样本：

| 步骤 | 说明 |
| --- | --- |
| 内部抽样 | 从 `train.parquet` 按 `label × 时间层` 抽取 500,000 条 |
| 内部拆分 | 每个时间层内部按时间顺序 8:2 切分 |
| 不平衡处理 | 内部训练集 SMOTE 到约 10% 正样本，内部验证集保持原始分布 |
| 调参目标 | 内部验证集 `PR-AUC` |
| 最终训练 | 定版后在 `train_smote_r10.parquet` 上训练 |
| 最终评估 | 在完整 `val.parquet` 上评估 |

阈值分析中，模型 artifact 保存的是内部验证集选择的阈值；最终验证集上的多阈值扫描只作为诊断，不用于重新选择阈值。

### 1.2 Baseline 结果

Baseline 模型在最终验证集上的结果如下：

| 模型 | ROC-AUC | PR-AUC | Log Loss | Precision@0.5 | Recall@0.5 | F1@0.5 | 内部选择阈值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM baseline | 0.992579 | 0.124940 | 0.011659 | 0.129056 | 0.472711 | 0.202756 | 0.40 |
| XGBoost baseline | 0.992698 | 0.124751 | 0.014061 | 0.104863 | 0.571303 | 0.177201 | 0.50 |
| Logistic Regression baseline | 0.979114 | 0.070285 | 0.055673 | 0.053911 | 0.761444 | 0.100693 | 0.90 |

Baseline 中 LightGBM 的 PR-AUC 和 F1@0.5 略高，因此它是未调优传统模型中的最强基准。

### 1.3 Optuna 调优结果

Optuna 设置：

| 项目 | LightGBM | XGBoost |
| --- | ---: | ---: |
| trials | 20 | 20 |
| 调参目标 | 内部验证集 PR-AUC | 内部验证集 PR-AUC |
| 内部训练集 | 400,000，SMOTE 后约 443,883 | 400,000，SMOTE 后约 443,883 |
| 内部验证集 | 100,000，保持原始分布 | 100,000，保持原始分布 |
| 最终训练集 | `train_smote_r10.parquet` | `train_smote_r10.parquet` |
| 最终验证集 | `val.parquet` | `val.parquet` |

Optuna 最优参数在最终验证集上的表现：

| 模型 | ROC-AUC | PR-AUC | Log Loss | Precision@0.5 | Recall@0.5 | F1@0.5 | 内部选择阈值 | 内部阈值在最终验证集 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM Optuna | 0.992444 | 0.105025 | 0.031937 | 0.056818 | 0.893486 | 0.106842 | 0.80 | 0.187271 |
| XGBoost Optuna | 0.993027 | 0.155966 | 0.005864 | 0.209866 | 0.220951 | 0.215266 | 0.20 | 0.211546 |

与 baseline 对比：

| 模型 | Baseline PR-AUC | Optuna PR-AUC | 变化 | 结论 |
| --- | ---: | ---: | ---: | --- |
| LightGBM | 0.124940 | 0.105025 | -0.019915 | 调优后下降，不采用 Optuna LightGBM |
| XGBoost | 0.124751 | 0.155966 | +0.031215 | 调优后明显提升，采用 Optuna XGBoost |

结论：XGBoost Optuna 是传统机器学习部分的最强单模型。LightGBM Optuna 虽然在内部验证集有较高分数，但在最终验证集泛化不如 baseline，因此后续融合保留 LightGBM baseline。

## 2. Tree 融合实验

对应脚本：

```text
src/run_tree_stacking.py
```

### 2.1 融合输入与训练方式

Tree 融合阶段使用上一部分筛选后的两个树模型：

| 模型 | 使用版本 | 路径 |
| --- | --- | --- |
| LightGBM | baseline | `output/baseline_models/lightgbm_baseline.joblib` |
| XGBoost | Optuna tuned | `output/optuna_tuned_models/xgboost_optuna.joblib` |

融合训练方式：

| 项目 | 设置 |
| --- | --- |
| OOF 样本 | 从 `train.parquet` 分层抽取 500,000 条 |
| OOF 正样本数 | 626 |
| folds | 5 |
| fold 内不平衡处理 | 只对 fold 训练部分 SMOTE 到 10% 正样本 |
| fold holdout | 保持原始分布 |
| 二层模型 | LogisticRegression，无 class_weight |
| 二层特征 | LightGBM 与 XGBoost 预测概率的 logit 变换 |
| 最终验证集 | `val.parquet`，不参与二层训练 |

### 2.2 Tree 融合结果

完整验证集结果：

| 模型/融合方案 | ROC-AUC | PR-AUC | Log Loss | Precision@阈值 | Recall@阈值 | F1@阈值 | 使用阈值 | 验证集最佳阈值 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM baseline | 0.992579 | 0.124940 | 0.011659 | 0.129056 | 0.472711 | 0.202756 | 0.50 | 0.210768 |
| XGBoost Optuna | 0.993027 | 0.155966 | 0.005864 | 0.209866 | 0.220951 | 0.215266 | 0.50 | 0.241669 |
| Tree equal | 0.993056 | 0.154567 | 0.008043 | 0.173913 | 0.316901 | 0.224579 | 0.50 | 0.230626 |
| OOF Stacking | 0.993160 | 0.152914 | 0.004837 | 0.121135 | 0.586268 | 0.200784 | 0.10 | 0.231124 |

从主指标 PR-AUC 看：

| 模型/融合方案 | PR-AUC |
| --- | ---: |
| XGBoost Optuna | 0.155966 |
| Tree equal | 0.154567 |
| OOF Stacking | 0.152914 |

结论：Tree equal 和 OOF Stacking 都没有超过 XGBoost Optuna 单模型的 PR-AUC。因此本轮 tree 融合没有带来主指标提升。OOF Stacking 的 ROC-AUC 和 Log Loss 略优，说明二层模型对概率排序和校准有一定帮助，但在极度不平衡任务中，最终仍以 PR-AUC 为主要选择依据。

## 3. DIN 调优与 DIN + Tree 融合

对应脚本：

```text
src/run_part7_din_stacking.py
```

### 3.1 DIN 调优流程

DIN 使用原始 `train.parquet`，不使用 SMOTE。调参阶段同样从训练集中分层抽取 500,000 条，并在每个时间层内部按时间顺序 8:2 切分。类别不平衡通过 `pos_weight` 处理，历史行为序列只使用候选样本发生时间之前的行为，避免序列特征使用未来信息。

DIN 调参比较 4 组配置，最终选择 `din_standard_adamw`：

| 参数 | 取值 |
| --- | ---: |
| embedding_dim | 32 |
| behavior_embedding_dim | 8 |
| hidden_size | 64 |
| num_layers | 1 |
| dropout | 0.2 |
| attention_heads | 0 |
| optimizer | AdamW |
| learning_rate | 0.001 |
| weight_decay | 0.00001 |
| use_pos_weight | True |
| fixed epochs | 5 |

### 3.2 DIN + Tree 固定权重预实验

在 DIN 全量训练版本上，进行了 DIN 与树模型的固定权重融合预实验。为保证和第 4 节 DIN + XGBoost 实验的控制变量一致，本节与第 4 节使用完全相同的 DIN 预测文件：

```text
output/part7_din_stacking/din_full_train_validation_predictions.csv
```

树模型口径为 `LightGBM baseline + XGBoost Optuna`，其中 LightGBM 预测来自 `output/baseline_models/lightgbm_baseline.joblib`，XGBoost 预测来自 `output/optuna_tuned_models/xgboost_optuna.joblib`。`--fusion-only` 模式复用了 `output/tree_stacking/stacking_validation_predictions.csv` 中的树模型验证集预测，避免重新调用底层树模型预测。

对应输出：

```text
output/part7_din_stacking/stacking_comparison_full_validation.csv
```

结果如下：

| 融合方案 | ROC-AUC | PR-AUC | Log Loss | F1@0.5 | 最佳阈值 | 最佳阈值 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 树模型等权 | 0.993056 | 0.154567 | 0.008043 | 0.224579 | 0.45 | 0.230626 |
| 树模型 + DIN 各 1/3 | 0.939454 | 0.123483 | 0.049756 | 0.216908 | 0.50 | 0.216908 |
| 树模型 + DIN，DIN 20% | 0.964611 | 0.151734 | 0.031600 | 0.229805 | 0.50 | 0.229805 |
| 树模型 + DIN，DIN 40% | 0.925064 | 0.106790 | 0.059769 | 0.203443 | 0.50 | 0.203443 |

结论：在统一使用全量 DIN 的控制变量口径下，DIN 20% 固定权重融合的 PR-AUC 为 0.151734，低于树模型等权的 0.154567，也低于 XGBoost Optuna 单模型的 0.155966。DIN 各 1/3 和 DIN 40% 的表现更低。
同时需要注意，DIN 20% 在默认阈值下的 F1 为 0.229805，略高于树模型等权的 0.224579，但 PR-AUC、ROC-AUC 和 Log Loss 均没有改善。因此在统一 DIN 版本后，当前没有证据证明 DIN 能提升 tree fusion 的主指标。

## 4. DIN 与 XGBoost 融合实验

对应脚本：

```text
src/run_xgb_din_fusion.py
```

### 4.1 实验设计

由于第 2 节中 Tree equal 和 OOF Stacking 都没有超过 XGBoost Optuna，当前最强基准是 XGBoost Optuna 单模型。因此新增一个更直接的实验：以 XGBoost Optuna 作为基准，测试 XGBoost 与全量训练 DIN 的固定权重融合，判断 DIN 是否能给最强单模型提供增益。

实验设置：

| 项目 | 设置 |
| --- | --- |
| XGBoost 输入 | `output/optuna_tuned_models/xgboost_optuna.joblib` |
| DIN 输入 | `output/part7_din_stacking/din_full_train_validation_predictions.csv` |
| 验证集 | `output/val.parquet` |
| DIN 权重 | 5%、10%、20%、30%、50% |
| 融合方式 | 固定概率加权，同时输出 rank 加权诊断 |
| 权重选择 | 预设权重，不在验证集上优化 |

概率融合公式：

```text
P_fusion = (1 - w) × P_XGBoost + w × P_DIN
```

其中 `w` 为 DIN 权重。Rank 融合只用于诊断 DIN 是否提供额外排序信息，不作为可部署模型结论。

### 4.2 XGBoost + DIN 结果

主要概率融合结果如下：

| 模型/融合方案 | ROC-AUC | PR-AUC | Log Loss | F1@0.5 | 相对 XGBoost PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost Optuna | 0.993027 | 0.155966 | 0.005864 | 0.215266 | 基准 |
| XGBoost 95% + DIN 5% | 0.970184 | 0.154236 | 0.011425 | 0.213694 | -0.001729 |
| XGBoost 90% + DIN 10% | 0.957563 | 0.149867 | 0.017277 | 0.213781 | -0.006099 |
| XGBoost 80% + DIN 20% | 0.935667 | 0.135907 | 0.029761 | 0.221124 | -0.020058 |
| XGBoost 70% + DIN 30% | 0.916905 | 0.119217 | 0.043390 | 0.228726 | -0.036749 |
| XGBoost 50% + DIN 50% | 0.881547 | 0.084203 | 0.075117 | 0.177642 | -0.071763 |
| DIN 全量训练 | 0.694346 | 0.004543 | 0.256045 | 0.007603 | -0.151422 |

所有 XGBoost + DIN 概率融合方案的 PR-AUC 都低于 XGBoost Optuna 单模型。最接近的是 DIN 5% 的轻量融合，PR-AUC 为 0.154236，仍低于 XGBoost Optuna 的 0.155966。

Rank 加权诊断同样没有超过 XGBoost Optuna，说明问题不只是 DIN 概率校准差，而是 DIN 当前排序信息本身也没有给 XGBoost 提供有效互补。

结论：以最强单模型 XGBoost Optuna 为基准时，DIN 没有带来 PR-AUC 增益。因此当前 DIN 不建议加入最终融合模型。

## 5. 最终模型选择对比

在完成传统机器学习调优、Tree Stacking、DIN + Tree 融合和 DIN + XGBoost 融合后，最终需要在“最强单模型”和“最佳固定权重融合方案”之间做选择。

| 模型/方案 | ROC-AUC | PR-AUC | Log Loss | F1@0.5 | 最佳阈值 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost Optuna | 0.993027 | 0.155966 | 0.005864 | 0.215266 | 0.241669 |
| LightGBM baseline 40% + XGBoost Optuna 40% + DIN 20% | 0.964611 | 0.151734 | 0.031600 | 0.229805 | 0.229805 |

从主指标 PR-AUC 看，三模型固定权重融合方案没有超过 XGBoost Optuna 单模型：

```text
PR-AUC 变化 = 0.151734 - 0.155966 = -0.004232
相对变化约 = -2.71%
```

这说明在控制 DIN 训练数据一致后，DIN 以 20% 权重加入 `LightGBM baseline + XGBoost Optuna` 没有提升正样本排序能力。

但该融合方案也存在权衡：

| 维度 | 更优方案 | 说明 |
| --- | --- | --- |
| PR-AUC | XGBoost Optuna | 单模型主指标更高 |
| ROC-AUC | XGBoost Optuna | 单模型整体排序更好 |
| Log Loss | XGBoost Optuna | 单模型概率校准更好 |
| 默认阈值 F1 | DIN 20% 三模型融合 | 0.229805 略高于 0.224579，但不是主指标 |

因此，若以 PR-AUC 作为最终模型选择标准，推荐采用：

```text
XGBoost Optuna
```

同时需要说明，DIN + Tree 和 DIN + XGBoost 两组实验现在均使用全量训练后的同一个 DIN 预测文件；三模型固定权重方案是消融实验，没有在验证集上自动搜索权重。

## 6. 最终结论

1. 传统机器学习调优流程已修正为内部验证调参，`val.parquet` 只用于最终评估。
2. XGBoost Optuna 是当前最强单模型，最终验证集 PR-AUC 为 0.155966。
3. LightGBM Optuna 的 PR-AUC 为 0.105025，低于 LightGBM baseline 的 0.124940，因此后续融合保留 LightGBM baseline。
4. Tree equal 和 OOF Stacking 没有超过 XGBoost Optuna 的 PR-AUC，因此单纯 tree 融合不替代最强单模型。
5. DIN 全量训练后 ROC-AUC 有提升，但 PR-AUC 仅 0.004543，显著低于树模型。
6. 在统一使用全量 DIN 的控制变量实验中，DIN 20% 三模型融合的 PR-AUC 为 0.151734，没有超过 Tree Equal 的 0.154567。
7. `XGBoost Optuna + 全量 DIN` 的所有固定权重方案也没有超过 XGBoost Optuna，说明当前 DIN 没有稳定的主指标增益。
8. 最终推荐以 XGBoost Optuna 作为主模型；DIN + Tree 和 DIN + XGBoost 保留为控制变量一致的消融实验。

## 7. 关联产出物

| 类型 | 路径 |
| --- | --- |
| Baseline 训练脚本 | `src/run_baseline_models.py` |
| Baseline 指标 | `output/baseline_models/baseline_metrics.csv` |
| Baseline 阈值分析 | `output/baseline_threshold_analysis/traditional_selected_threshold_metrics.csv` |
| Optuna 调优脚本 | `src/run_optuna_tuned_models.py` |
| Optuna 指标 | `output/optuna_tuned_models/optuna_tuned_metrics.csv` |
| 最强 XGBoost 模型 | `output/optuna_tuned_models/xgboost_optuna.joblib` |
| 保留的 LightGBM 模型 | `output/baseline_models/lightgbm_baseline.joblib` |
| Tree Stacking 脚本 | `src/run_tree_stacking.py` |
| Tree Stacking 指标 | `output/tree_stacking/stacking_validation_metrics.csv` |
| DIN 训练脚本 | `src/run_part7_din_stacking.py` |
| DIN 全量模型 | `output/part7_din_stacking/din_final_full_train/din_baseline.pt` |
| DIN 全量指标 | `output/part7_din_stacking/din_full_train_metrics.json` |
| DIN + Tree 融合结果 | `output/part7_din_stacking/stacking_comparison_full_validation.csv` |
| XGBoost + DIN 融合脚本 | `src/run_xgb_din_fusion.py` |
| XGBoost + DIN 融合指标 | `output/xgb_din_fusion/xgb_din_fusion_metrics.csv` |
| XGBoost + DIN 融合报告 | `output/xgb_din_fusion/xgb_din_fusion_report.md` |
