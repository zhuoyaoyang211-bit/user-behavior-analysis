# 淘宝用户行为分析 (user-behavior-analysis)

基于阿里天池淘宝用户行为公开数据集（**1万用户采样版**）的端到端数据分析项目，按"数据清洗 → 中间表设计 → 特征工程"三阶段完成。

> 原始数据：12,256,906 行用户行为记录，1 万独立用户，287 万独立商品，1 个月时间窗口（2025-11-18 ~ 2025-12-18）。

---

## 目录结构

```
项目1/
├── src/                          # 源代码
│   ├── config.py                 # 路径与全局配置
│   ├── main.py                   # 入口：跑完整 Part1 流程
│   ├── data_loader.py            # 原始数据加载
│   ├── data_cleaner.py           # 数据清洗逻辑
│   ├── data_quality.py           # 数据质量评估
│   ├── build_dims.py             # 入口：跑 Part2 中间表构建
│   ├── build_features.py         # 入口：跑 Part3 特征宽表构建
│   ├── common/                   # 通用工具（异常/日志/常量/聚合规范）
│   ├── dim_builders/             # Part2 中间表 builder（user/item/category/time）
│   └── feature_engineering/      # Part3 特征 builder（业务/生命周期/语义）
│
├── output/                       # 产出物（按 Part 拆分）
│   ├── cleaned_data.parquet          # Part1 清洗后的明细
│   ├── quality_report.txt            # Part1 数据质量报告
│   ├── dim_user.parquet              # Part2 用户维表
│   ├── dim_item.parquet              # Part2 商品维表
│   ├── dim_category.parquet          # Part2 类目维表
│   ├── dim_time_hourly.parquet       # Part2 时间维表（小时）
│   ├── dim_time_daily.parquet        # Part2 时间维表（天）
│   ├── dim_time_weekday_hour.parquet # Part2 时间维表（星期×小时）
│   └── feature_wide_table.parquet    # Part3 特征宽表（468 万 × 47 列）
│
├── docs/                         # 设计文档
│   ├── Part1_代码详解.md
│   ├── Part2_中间表设计.md
│   └── Part3_特征字典.md
│
├── user_behavior_processed.csv   # 原始数据（1 万用户采样，469MB）
├── 用户行为分析背景大纲.pdf       # 项目需求大纲
├── week_2周报.docx                # 第 2 周周报
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 快速开始

### 1. 环境

- Python 3.10+（推荐 3.13）
- 安装依赖：`pip install -r requirements.txt`
- 主要库：pandas、pyarrow、scikit-learn、scipy

### 2. 跑 Part1（数据清洗）

```bash
cd src
python main.py
```

产出：`output/cleaned_data.parquet` + `output/quality_report.txt`

### 3. 跑 Part2（中间表）

```bash
cd src
python build_dims.py
```

产出：6 张 dim_*.parquet 中间表（用户/商品/类目/时间）

### 4. 跑 Part3（特征宽表）

```bash
cd src
python build_features.py
```

产出：`output/feature_wide_table.parquet`（468 万行 × 47 列特征）

---

## 阶段产出速览

| Part | 关键产出 | 行数 × 列数 | 文档 |
|------|---------|------------|------|
| Part1 数据清洗 | cleaned_data.parquet | 12,256,906 × 6 | docs/Part1_代码详解.md |
| Part2 中间表设计 | 6 张 dim 表 | 用户1万×17 / 商品287万×11 / 类目8916×11 / 时间744+31+168 | docs/Part2_中间表设计.md |
| Part3 特征工程 | feature_wide_table.parquet | 4,686,904 × 47 | docs/Part3_特征字典.md |

### Part3 47 个特征分类

| 类别 | 数量 | 字段 |
|------|-----|------|
| 行为计数 | 4 | user_pv/fav/cart/buy_count |
| 活跃度 | 4 | active_days, day_pct, evening_pct, night_pct |
| 购买行为 | 5 | buy_item_count, buy_conversion_rate, fav_to_buy_rate, cart_to_buy_rate, repurchase_item_count |
| 重度用户标记 | 1 | is_power_user |
| 用户级辅助 | 2 | user_streak_days, user_avg_interval_hours |
| 商品热度趋势 | 1 | item_decay_slope |
| 购买路径 | 1 | buy_path_type (0-4) |
| RFM 评分 | 3 | rfm_r_score, rfm_f_score, rfm_m_score |
| 偏好/SVD | 2 | user_category_pref_score, user_item_svd_score |
| 主键列 | 3 | user_id, item_id, item_category |
| 商品维表特征 | 9 | item_pv_count, item_fav_count, item_cart_count, ... |
| 类目维表特征 | 9 | cat_item_count, cat_buy_item_count, ... |

---

## 核心设计决策

完整设计思路见 `docs/` 下三份文档。几条关键：

- **Part1 重复值不删除**：原始数据四元组重复率 49.31%，经老师确认是真实用户行为（如反复浏览），全部保留
- **Part1 爬虫检测**：0 个爬虫，733 名重度买家打 `is_power_user` 标记
- **Part2 转化率漏斗口径**：转化率 = 做了上游行为且购买的对数 ÷ 做了上游行为的对数（值域 [0,1]）
- **Part3 主键全量**：以 (user_id, item_id) 为粒度，共 468 万对，**不限于购买用户**
- **Part3 buy_path_type 对级**：路径分类按 (用户,商品) 对算，0=未买/1=直接买/2=收藏后买/3=加购后买/4=收藏+加购后买
- **Part3 RFM 观察日 = 2025-12-19**：取数据最后一天结束时刻，避免 12-18 当天购买用户 R 值变负

---

## License

项目数据来自阿里天池公开数据集，仅供学习使用。
