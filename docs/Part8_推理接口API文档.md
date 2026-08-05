# Part8 本地批量推理接口 API 文档

## 1. 文档概述

本文档说明 `src/run_batch_inference.py` 的调用方式、输入输出协议、参数定义、异常处理和运行示例。

当前实现是**本地命令行批量推理接口**，不是 HTTP/REST 服务。接口接收原始用户行为数据文件，自动完成数据清洗、特征计算、特征预处理和 XGBoost 模型预测，最终输出用户-商品购买概率。

核心代码：

| 文件 | 职责 |
| --- | --- |
| `src/run_batch_inference.py` | 推理入口、参数解析、模型加载、文件读写、预测结果输出 |
| `src/inference_feature_builder.py` | 原始行为数据的分块聚合和推理特征构建 |
| `output/optuna_tuned_models/xgboost_optuna.joblib` | Optuna 优化后的 XGBoost 模型 |
| `output/feature_wide_table.parquet` | 训练阶段特征宽表，用于拟合推理预处理参数 |
| `output/inference/xgboost_inference_preprocessor.joblib` | 缓存后的推理预处理参数 |

## 2. 接口基本信息

| 项目 | 内容 |
| --- | --- |
| 接口名称 | XGBoost 本地批量推理接口 |
| 调用方式 | Python CLI |
| 入口脚本 | `src/run_batch_inference.py` |
| 输入 | 原始行为数据 CSV/Parquet 文件 |
| 输出 | 购买概率预测 CSV + 推理元信息 JSON |
| 默认模型 | `output/optuna_tuned_models/xgboost_optuna.joblib` |
| 默认阈值 | 从模型 artifact 的 `selected_threshold` 读取，当前为 `0.2` |
| 默认候选范围 | 原始行为数据中出现过的全部 `(user_id, item_id)` 对 |

## 3. 调用语法

```bash
cd /Users/yangzhuoyao/Desktop/阿里/项目1

python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --output-dir output/inference/xgboost_optuna_batch
```

执行成功后，控制台会输出预测结果文件路径：

```text
Batch inference finished: output/inference/xgboost_optuna_batch/purchase_probability_predictions.csv
```

## 4. 命令参数

| 参数 | 类型 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- | --- |
| `--raw-path` | `Path` | `user_behavior_processed.csv` | 否 | 原始行为数据路径，支持 `.csv`、`.parquet`、`.pq` |
| `--model-path` | `Path` | `output/optuna_tuned_models/xgboost_optuna.joblib` | 否 | XGBoost 模型 artifact 路径 |
| `--output-dir` | `Path` | `output/inference/xgboost_optuna_batch` | 否 | 推理结果输出目录 |
| `--output-name` | `str` | `purchase_probability_predictions.csv` | 否 | 预测结果文件名 |
| `--candidate-path` | `Path` | 无 | 否 | 可选候选用户-商品对文件；不传时预测原始行为中所有用户-商品对 |
| `--reference-wide-path` | `Path` | `output/feature_wide_table.parquet` | 否 | 训练特征宽表，用于拟合预处理参数 |
| `--preprocessor-path` | `Path` | `output/inference/xgboost_inference_preprocessor.joblib` | 否 | 推理预处理参数缓存路径 |
| `--rebuild-preprocessor` | Flag | 关闭 | 否 | 强制重新扫描参考宽表并生成预处理缓存 |
| `--threshold` | `float` | 模型 artifact 中的 `selected_threshold` | 否 | 生成 `prediction_label` 时使用的分类阈值 |
| `--batch-size` | `int` | `500000` | 否 | 每批送入 XGBoost 预测的候选行数 |
| `--chunk-size` | `int` | `1000000` | 否 | 原始数据和参考宽表的分块读取行数 |
| `--candidate-failure-policy` | `error/drop` | `error` | 否 | 候选商品无法匹配类目时的处理方式 |
| `--limit-rows` | `int` | 无 | 否 | 仅读取前 N 行，用于接口冒烟测试 |

参数约束：

- `--chunk-size` 必须大于 0。
- `--batch-size` 必须大于 0。
- `--threshold` 建议取 `[0, 1]` 区间内的值。
- `--candidate-failure-policy` 只能取 `error` 或 `drop`。

## 5. 输入数据协议

### 5.1 原始行为数据

原始行为文件必须包含以下字段：

| 字段 | 类型 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `time` | 字符串/时间 | 是 | 行为时间，推荐格式 `YYYY-MM-DD HH` |
| `user_id` | 整数 | 是 | 用户唯一标识 |
| `item_id` | 整数 | 是 | 商品唯一标识 |
| `item_category` | 整数 | 是 | 商品类目标识 |
| `behavior_type` | 整数 | 是 | 行为类型，只允许 `1/2/3/4` |

行为类型定义：

| `behavior_type` | 含义 |
| ---: | --- |
| `1` | 浏览 |
| `2` | 收藏 |
| `3` | 加购 |
| `4` | 购买 |

CSV 示例：

```csv
time,user_id,item_id,item_category,behavior_type
2025-12-06 02,98047837,232431562,4245,1
2025-12-09 20,97726136,383583590,5894,1
2025-12-18 11,98607707,64749712,2883,1
```

清洗规则：

1. 校验必要字段。
2. 校验同一 `item_id` 是否对应多个 `item_category`。
3. 保留四元组重复记录，不擅自删除业务行为。
4. 过滤非法行为类型。
5. 标准化 `time`，无法解析的时间记录会被移除。
6. 按块累计用户、商品、类目、购买路径、时间间隔和趋势特征。

### 5.2 候选用户-商品对文件

`--candidate-path` 为可选参数。候选文件支持 CSV 和 Parquet，至少包含：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `user_id` | 是 | 用户唯一标识 |
| `item_id` | 是 | 商品唯一标识 |
| `item_category` | 否 | 商品类目；不提供时从原始行为数据匹配 |

CSV 示例：

```csv
user_id,item_id
98047837,232431562
97726136,383583590
```

候选处理规则：

- 同一 `(user_id, item_id)` 重复候选会去重。
- 候选文件提供的 `item_category` 与原始行为数据不一致时，接口直接报错。
- 候选商品无法从原始行为数据匹配到 `item_category` 时：
  - `error`：保存失败明细后终止推理；
  - `drop`：保存失败明细后丢弃该候选，继续推理。
- 失败明细文件为：

```text
<output-dir>/candidate_rejections.csv
```

未传 `--candidate-path` 时，接口默认预测原始行为中出现过的所有用户-商品对。

## 6. 推理处理流程

```text
原始 CSV/Parquet
        |
        v
分块读取
        |
        v
数据清洗与时间标准化
        |
        v
增量特征聚合
  用户 / 商品 / 类目
  购买路径 / RFM
  商品热度趋势
  用户平均行为间隔
        |
        v
候选用户-商品对匹配
        |
        v
加载或生成预处理参数
        |
        v
按训练特征列顺序转换
        |
        v
分批 XGBoost predict_proba
        |
        v
购买概率 CSV + 元信息 JSON
```

模型输入特征由 model artifact 中的 `feature_cols` 决定。当前优化后 XGBoost 使用 24 个特征：

```text
item_pv_count
item_fav_count
item_cart_count
item_buy_count
item_buy_user_count
item_pv_to_buy_rate
item_cart_to_buy_rate
item_repurchase_user_count
cat_pv_count
cat_view_user_count
user_pv_count
day_pct
night_pct
buy_conversion_rate
fav_to_buy_rate
cart_to_buy_rate
repurchase_item_count
item_decay_slope
user_avg_interval_hours
rfm_r_score
rfm_f_score
rfm_m_score
user_category_pref_score
item_category_te
```

推理阶段不会重新用待预测数据拟合标准化参数。首次运行会根据 `--reference-wide-path` 生成缓存，后续运行直接加载：

```text
output/inference/xgboost_inference_preprocessor.joblib
```

只有以下情况需要使用 `--rebuild-preprocessor`：

- 更换了最终模型的特征列；
- 更换了训练特征宽表；
- 修改了训练阶段的缺失值填充或标准化逻辑；
- 预处理缓存被删除或损坏。

## 7. 输出数据协议

### 7.1 预测结果 CSV

默认文件：

```text
<output-dir>/purchase_probability_predictions.csv
```

字段定义：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `user_id` | 整数 | 用户标识 |
| `item_id` | 整数 | 商品标识 |
| `item_category` | 整数 | 商品类目 |
| `purchase_probability` | 浮点数 | XGBoost 输出的购买概率，范围通常为 `[0, 1]` |
| `prediction_label` | `0/1` | 概率大于等于阈值时为 `1`，否则为 `0` |
| `threshold` | 浮点数 | 本次推理实际使用的阈值 |
| `rank_in_user` | 整数 | 同一用户内部按照购买概率降序排列的排名 |

示例：

```csv
user_id,item_id,item_category,purchase_probability,prediction_label,threshold,rank_in_user
10131505,280341937,1863,0.210085,1,0.2,1
10131505,25530939,2130,0.000001,0,0.2,2
```

注意：

- `purchase_probability` 是模型概率，不是购买数量。
- `prediction_label` 受阈值影响；修改 `--threshold` 不会重新训练模型。
- `rank_in_user` 只表示候选结果在当前批次内的用户内排序。

### 7.2 推理元信息 JSON

默认文件：

```text
<output-dir>/inference_metadata.json
```

主要字段：

| 字段 | 说明 |
| --- | --- |
| `raw_path` | 原始行为数据路径 |
| `candidate_path` | 候选文件路径；未使用时为 `null` |
| `model_path` | 模型 artifact 路径 |
| `reference_wide_path` | 预处理参考宽表路径 |
| `preprocessor_path` | 预处理缓存路径 |
| `output_path` | 预测结果路径 |
| `memory_mode` | 当前内存处理模式 |
| `chunk_size` | 原始数据读取块大小 |
| `batch_size` | 模型预测块大小 |
| `candidate_failure_policy` | 候选失败处理策略 |
| `raw_rows` | 原始读取行数 |
| `cleaned_rows` | 清洗后行数 |
| `input_chunk_count` | 输入分块数量 |
| `candidate_stats` | 候选总数、去重数、失败数 |
| `prediction_rows` | 最终预测行数 |
| `feature_cols` | 本次模型使用的特征列 |
| `threshold` | 本次推理阈值 |
| `positive_predictions` | 预测标签为 1 的数量 |
| `elapsed_seconds` | 推理耗时 |
| `random_state` | 项目随机种子 |

## 8. 调用示例

### 8.1 全量原始行为推理

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --output-dir output/inference/xgboost_optuna_batch
```

### 8.2 内存较小的电脑

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --output-dir output/inference/xgboost_optuna_batch \
  --chunk-size 200000 \
  --batch-size 100000
```

### 8.3 指定候选用户-商品对

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --candidate-path input/candidates.csv \
  --output-dir output/inference/candidate_batch
```

### 8.4 允许丢弃无法匹配类目的候选

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --candidate-path input/candidates.csv \
  --candidate-failure-policy drop \
  --output-dir output/inference/candidate_batch
```

### 8.5 修改分类阈值

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --threshold 0.10 \
  --output-dir output/inference/threshold_010
```

### 8.6 首次生成或强制刷新预处理缓存

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --rebuild-preprocessor \
  --output-dir output/inference/xgboost_optuna_batch
```

### 8.7 小样本冒烟测试

```bash
python src/run_batch_inference.py \
  --raw-path user_behavior_processed.csv \
  --limit-rows 5000 \
  --chunk-size 1000 \
  --batch-size 1000 \
  --output-dir output/inference/smoke_test
```

## 9. 异常与排查

| 异常现象 | 常见原因 | 处理方式 |
| --- | --- | --- |
| `Input file does not exist` | 输入路径错误 | 检查 `--raw-path` 或 `--candidate-path` |
| `Unsupported file suffix` | 文件不是 CSV/Parquet | 修改文件后缀或转换格式 |
| `Behavior chunk missing columns` | 原始行为数据缺少必要字段 | 补齐 `time/user_id/item_id/item_category/behavior_type` |
| `item_id maps to multiple item_category values` | 同一商品关联多个类目 | 清理原始数据后重新推理 |
| `candidate rows cannot match item_category` | 候选商品不在原始行为数据中 | 修正候选文件，或显式使用 `drop` |
| `Cached preprocessor feature_cols differs from model artifact` | 模型和预处理缓存不匹配 | 删除旧缓存或使用 `--rebuild-preprocessor` |
| `batch_size must be positive` | 批大小配置非法 | 设置大于 0 的整数 |
| 内存不足 | 分块/批次过大，或候选规模过大 | 降低 `--chunk-size` 和 `--batch-size` |

接口失败时会抛出异常并以非 0 状态退出；成功时以 0 状态退出并生成预测 CSV。

## 10. 内存与性能说明

当前版本采用三层控制：

1. 原始 CSV/Parquet 按 `--chunk-size` 分块读取。
2. 特征工程使用增量聚合，不拼接完整原始行为明细。
3. XGBoost 使用 `--batch-size` 分批生成概率。

需要注意：

- 聚合后的用户-商品对、用户小时记录和趋势统计仍需在内存中保留，这是构建当前特征的必要状态。
- 候选文件目前会整体读取，候选文件特别大时应先拆分文件分批调用。
- `--rebuild-preprocessor` 会分块扫描训练宽表多次，首次运行耗时会明显增加；通常只需执行一次。
- 建议先使用 `--limit-rows 5000` 完成冒烟测试，再运行全量推理。

## 11. 当前接口边界

当前接口支持：

- 本地 CSV/Parquet 文件输入；
- 本地批量预测；
- 自定义模型、阈值、候选对和内存参数；
- 预测结果和运行元信息落盘。

当前接口不直接提供：

- HTTP REST endpoint；
- 在线单条 JSON 请求；
- 数据库读取；
- 自动上传对象存储；
- 多模型融合预测。

如果后续需要对外提供 REST API，应在本脚本外增加 FastAPI/Flask 服务层，将请求数据转换为临时输入文件或 DataFrame，再调用同一套特征聚合和模型预测逻辑。

## 12. 产出物清单

| 产出物 | 默认路径 | 说明 |
| --- | --- | --- |
| 预测结果 | `output/inference/xgboost_optuna_batch/purchase_probability_predictions.csv` | 用户-商品购买概率和标签 |
| 推理元信息 | `output/inference/xgboost_optuna_batch/inference_metadata.json` | 输入、模型、参数、耗时和统计信息 |
| 预处理缓存 | `output/inference/xgboost_inference_preprocessor.joblib` | 目标编码、填充、标准化参数 |
| 候选失败明细 | `<output-dir>/candidate_rejections.csv` | 仅在候选匹配失败时生成 |

