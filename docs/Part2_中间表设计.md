# Part 2 中间表设计说明

## 1. 概述

### 1.1 目标

基于 Part1 产出的清洗数据（`cleaned_data.parquet`，12,256,906行），按用户、商品、时间三个维度进行预聚合，生成中间表层，支撑 Part3 的快速查询与特征拼接。

### 1.2 中间表清单

本步骤产出三类维度共6张中间表：

| 维度类别 | 表名 | 主键（索引列） | 数据列数 | 行数 |
|---|---|---|---|---|
| 用户 | dim_user | user_id | 16 | 10,000 |
| 商品（单品） | dim_item | item_id | 10 | 2,876,947 |
| 商品（类目） | dim_category | item_category | 10 | 8,916 |
| 时间（小时） | dim_time_hourly | date, hour | 6 | 744 |
| 时间（日） | dim_time_daily | date | 6 | 31 |
| 时间（周×小时） | dim_time_weekday_hour | weekday, hour | 6 | 168 |

### 1.3 代码结构

```
src/
├── common/agg_specs.py        聚合指标常量定义
├── dim_builders/
│   ├── user_dim_builder.py
│   ├── item_dim_builder.py
│   └── time_dim_builder.py
└── build_dims.py              主入口
```

---

## 2. dim_user — 用户维度表

### 2.1 字段定义

主键 `user_id` 存为 Parquet 索引列，以下为16个数据列：

| 字段 | 类型 | 说明 |
|---|---|---|
| pv_count | int64 | 浏览次数 |
| fav_count | int64 | 收藏次数 |
| cart_count | int64 | 加购次数 |
| buy_count | int64 | 购买次数 |
| active_days | int64 | 活跃天数（1~31） |
| first_active_time | datetime64[us] | 首次行为时间 |
| last_active_time | datetime64[us] | 末次行为时间 |
| day_pct | float64 | 白天行为占比（6-17点） |
| evening_pct | float64 | 晚间行为占比（18-23点） |
| night_pct | float64 | 深夜行为占比（0-5点） |
| buy_item_count | int32 | 购买商品数（按item_id去重） |
| buy_conversion_rate | float64 | 浏览→购买转化率 = 浏览且购买的商品数 ÷ 浏览过的商品数 |
| fav_to_buy_rate | float64 | 收藏→购买转化率 = 收藏且购买的商品数 ÷ 收藏过的商品数 |
| cart_to_buy_rate | float64 | 加购→购买转化率 = 加购且购买的商品数 ÷ 加购过的商品数 |
| repurchase_item_count | int32 | 复购商品数（购买≥2次的商品数） |
| is_power_user | bool | 是否重度买家（复用Part1标记） |

### 2.2 计算口径

- **pv口径**：按原始记录条数统计，含Part1保留的重复记录
- **uv口径**：按user_id或item_id去重统计
- **活跃度**：四种行为类型任意一种即计为活跃，不区分行为类型
- **时段划分**：白天6-17点 / 晚间18-23点 / 深夜0-5点
- **转化率（漏斗口径）**：分子是分母的子集（做了上游行为且最终购买），值域 [0, 1]，分母为0时填0

---

## 3. dim_item — 商品单品维度表

### 3.1 字段定义

主键 `item_id` 存为 Parquet 索引列，以下为10个数据列：

| 字段 | 类型 | 说明 |
|---|---|---|
| item_category | int16 | 所属类目（外键→dim_category） |
| pv_count | int64 | 浏览次数 |
| view_user_count | int32 | 浏览独立用户数 |
| fav_count | int64 | 收藏次数 |
| cart_count | int64 | 加购次数 |
| buy_count | int64 | 购买次数 |
| buy_user_count | int32 | 购买独立用户数 |
| pv_to_buy_rate | float64 | 浏览→购买转化率 = 浏览且购买的用户数 ÷ 浏览过的用户数 |
| cart_to_buy_rate | float64 | 加购→购买转化率 = 加购且购买的用户数 ÷ 加购过的用户数 |
| repurchase_user_count | int32 | 复购用户数（购买≥2次的用户数） |

### 3.2 设计说明

全量保留2,876,947个商品，不做活跃度过滤。尾部商品后续可用于长尾分布分析。

---

## 4. dim_category — 商品类目维度表

### 4.1 字段定义

主键 `item_category` 存为 Parquet 索引列，以下为10个数据列：

| 字段 | 类型 | 说明 |
|---|---|---|
| item_count | int64 | 该类目下商品总数 |
| buy_item_count | int32 | 有购买记录的商品数 |
| pv_count | int64 | 类目总浏览次数 |
| view_user_count | int32 | 类目总浏览独立用户数 |
| fav_count | int64 | 类目总收藏次数 |
| cart_count | int64 | 类目总加购次数 |
| buy_count | int64 | 类目总购买次数 |
| buy_user_count | int32 | 类目总购买独立用户数 |
| pv_to_buy_rate | float64 | 浏览→购买转化率 = 被浏览且被买的商品数 ÷ 被浏览过的商品数 |
| buy_item_pct | float64 | 动销率 = buy_item_count / item_count |

---

## 5. dim_time — 时间维度表

### 5.1 统一字段

三张时间表共享以下6个数据列：

| 字段 | 类型 | 说明 |
|---|---|---|
| pv_count | int64 | 浏览次数 |
| fav_count | int64 | 收藏次数 |
| cart_count | int64 | 加购次数 |
| buy_count | int64 | 购买次数 |
| active_user_count | int32 | 独立活跃用户数 |
| active_item_count | int32 | 独立活跃商品数 |

### 5.2 三张表的区别

| 表名 | 索引列 | 聚合方式 | 用途 |
|---|---|---|---|
| dim_time_hourly | date, hour | 按(日期,小时) | 小时级行为趋势 |
| dim_time_daily | date | 按日期 | 日级行为趋势 |
| dim_time_weekday_hour | weekday, hour | 按(星期几,小时) | 周期性时段规律 |

> weekday: 0=周一, 6=周日

---

## 6. 关联键与索引

### 6.1 关联关系

三张维度表通过各自主键与流水表关联，支持步骤三的多维交叉查询：

```
dim_user.user_id           ← 流水表.user_id
dim_item.item_id           ← 流水表.item_id
dim_category.item_category ← dim_item.item_category
dim_time_*.date/hour       ← 流水表.time
```

### 6.2 索引方案

每张表按主键 `sort + set_index` 后存储为 Parquet。Parquet 列式存储自带 min/max 索引，set_index 写入行组元数据，查询时可跳过不相关行组。

---

## 7. 与 Part1 的衔接

| Part1 产物 | Part2 使用方式 |
|---|---|
| cleaned_data.parquet | 唯一输入源 |
| is_power_user 标记 | dim_user 直接复用 |
| item_id→item_category 一对一校验 | dim_item 取 first 安全 |
| 四元组重复保留不删除 | 所有 pv 口径指标如实统计 |

---

## 8. 验收结果

| 检查项 | 结果 |
|---|---|
| 6张 parquet 全部生成 | ✓ |
| 行数：1万 / 287万 / 8916 / 744 / 31 / 168 | ✓ |
| 主键无重复 | ✓ |
| 行为量守恒（各维度加总=原始数据） | ✓ |
| 无缺失值 | ✓ |
