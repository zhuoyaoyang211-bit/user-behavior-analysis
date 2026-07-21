# Part5 类别不平衡处理方案对比报告

## 1. 报告定位

本报告对应步骤 5 中的类别不平衡处理方案对比。

根据老师建议，本次在 50% 平衡基线基础上，同时对 SMOTE 过采样和欠采样新增 10%、25% 正样本占比试点。

## 2. 实验前提

- 训练集：2,528,026 行，建模特征 26 列
- 原始正样本占比：0.1117%
- 验证集保持原始类别分布，不参与采样
- 模型：L1 逻辑回归，`C=1.0`，`solver=saga`
- 本地运行环境：MacBook Air M2，内存 8GB

### 2.1 随机种子设置说明

本次实验中，SMOTE 每个正样本占比运行 1 次，欠采样每个正样本占比运行 3 次（random_state=42/100/2024）后取平均。

这样设计的原因如下：

- 欠采样会从 252 万级负样本中随机抽取很少一部分负样本，随机抽到的负样本集合会明显影响模型结果，因此需要多次运行取平均，降低偶然性。
- SMOTE 不删除原始负样本，而是在正样本邻域中生成合成样本，主要随机性来自合成样本插值过程，随机波动通常小于欠采样。
- SMOTE 处理后的数据规模较大：10% 为 280.6 万行，25% 为 336.7 万行，50% 为 505.0 万行。在 8GB 内存的本地机器上，对 SMOTE 再做 3 个随机种子会显著增加训练时间和内存压力。

因此，本次采用“SMOTE 单次运行 + 欠采样多次运行取平均”的折中方案，在保证可运行性的同时重点控制欠采样的随机波动。

## 3. 不同正样本占比处理后数据集规模

### 3.1 SMOTE 过采样

| 目标正样本占比 | 处理后行数 | 实际正样本占比 | 输出文件 |
| --- | ---: | ---: | --- |
| 10% | 2,805,781 | 10.00% | `output/train_smote_r10.parquet` |
| 25% | 3,366,937 | 25.00% | `output/train_smote_r25.parquet` |
| 50% | 5,050,406 | 50.00% | `output/train_smote.parquet` |

### 3.2 欠采样

| 目标正样本占比 | random_state | 处理后行数 | 实际正样本占比 | 输出文件 |
| --- | ---: | ---: | ---: | --- |
| 10% | 42 | 28,230 | 10.00% | `output/train_undersample_rs42_r10.parquet` |
| 10% | 100 | 28,230 | 10.00% | `output/train_undersample_rs100_r10.parquet` |
| 10% | 2024 | 28,230 | 10.00% | `output/train_undersample_rs2024_r10.parquet` |
| 25% | 42 | 11,292 | 25.00% | `output/train_undersample_rs42_r25.parquet` |
| 25% | 100 | 11,292 | 25.00% | `output/train_undersample_rs100_r25.parquet` |
| 25% | 2024 | 11,292 | 25.00% | `output/train_undersample_rs2024_r25.parquet` |
| 50% | 42 | 5,646 | 50.00% | `output/train_undersample_rs42_r50.parquet` |
| 50% | 100 | 5,646 | 50.00% | `output/train_undersample_rs100_r50.parquet` |
| 50% | 2024 | 5,646 | 50.00% | `output/train_undersample_rs2024_r50.parquet` |

### 3.3 类别权重

- 类别权重不改变数据比例，只在损失函数层面平衡正负样本。
- 负样本权重：`0.5006`
- 正样本权重：`447.7552`
- 正/负权重比：`894.5×`

## 4. 50% 平衡基线三方案对比

| 方案 | Precision | Recall | F1 | AUC | TrainAcc | n_train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SMOTE 过采样 | 0.0306 | 0.9452 | 0.0592 | 0.9878 | 0.9627 | 5,050,406 |
| 类别权重 | 0.0303 | 0.9458 | 0.0588 | 0.9884 | 0.9613 | 2,528,026 |
| 欠采样 (n_runs=3) | 0.0298 | 0.9495 | 0.0578 | 0.9884 | 0.9610 | 5,646 |

## 5. SMOTE 不同正样本占比对比

SMOTE 每个比例运行 1 次。

| 方案 | Precision | Recall | F1 | AUC | TrainAcc | n_train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SMOTE10% | 0.0685 | 0.7824 | 0.1259 | 0.9863 | 0.9690 | 2,805,781 |
| SMOTE25% | 0.0469 | 0.8868 | 0.0892 | 0.9872 | 0.9578 | 3,366,937 |
| SMOTE50% | 0.0306 | 0.9452 | 0.0592 | 0.9878 | 0.9627 | 5,050,406 |

## 6. 欠采样不同正样本占比对比

欠采样每个比例运行 3 次，表中为平均值。

| 方案 | Precision | Recall | F1 | AUC | TrainAcc | n_train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 欠采样10% | 0.0698 | 0.7858 | 0.1282 | 0.9879 | 0.9685 | 28,230 |
| 欠采样25% | 0.0467 | 0.8854 | 0.0886 | 0.9884 | 0.9579 | 11,292 |
| 欠采样50% | 0.0298 | 0.9495 | 0.0578 | 0.9884 | 0.9610 | 5,646 |

注：欠采样指标为 3 次随机种子（42/100/2024）的平均值。

## 7. 结论

- 50% 平衡基线中 F1 最高：`SMOTE 过采样`（F1 = 0.0592）。
- 10%/25%/50% 比例试点中 F1 最高：`欠采样10%`（F1 = 0.1282）。
- SMOTE 与欠采样均已覆盖 10%、25%、50% 三档正样本占比。
- 类别权重不生成新数据、不删除样本，因此不参与正样本占比试点，只作为损失层面的平衡基线。
- 从结果看，10% 正样本占比相比 50% 平衡方案显著提升 Precision 和 F1；50% 平衡方案 Recall 更高，但误报更多。
- 受本地 8GB 内存限制，SMOTE 未做多随机种子重复实验；如迁移到更高内存环境，可进一步对 SMOTE 也做 3 次随机种子重复，验证结果稳定性。

## 8. 产出文件

| 文件 | 说明 |
| --- | --- |
| `output/train_smote_r10.parquet` | SMOTE 10% 正样本占比训练集 |
| `output/train_smote_r25.parquet` | SMOTE 25% 正样本占比训练集 |
| `output/train_smote.parquet` | SMOTE 50% 正样本占比训练集 |
| `output/train_undersample_rs42_r10.parquet` | 欠采样 10%，seed=42 |
| `output/train_undersample_rs100_r10.parquet` | 欠采样 10%，seed=100 |
| `output/train_undersample_rs2024_r10.parquet` | 欠采样 10%，seed=2024 |
| `output/train_undersample_rs42_r25.parquet` | 欠采样 25%，seed=42 |
| `output/train_undersample_rs100_r25.parquet` | 欠采样 25%，seed=100 |
| `output/train_undersample_rs2024_r25.parquet` | 欠采样 25%，seed=2024 |
| `output/train_undersample_rs42_r50.parquet` | 欠采样 50%，seed=42 |
| `output/train_undersample_rs100_r50.parquet` | 欠采样 50%，seed=100 |
| `output/train_undersample_rs2024_r50.parquet` | 欠采样 50%，seed=2024 |
| `output/imbalance_comparison.json` | 类别不平衡处理实验结果汇总 |
