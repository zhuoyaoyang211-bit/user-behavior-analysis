# Part3 特征字典

> 特征宽表 `output/feature_wide_table.parquet` 字段说明文档
>
> 主键：(user_id, item_id) | 规模：4,686,904 行 × 47 列 | 观察日：2025-12-18

## 1. 主键（2列）

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int32 | 用户ID |
| `item_id` | int32 | 商品ID |

## 2. 类目标识（1列）

| 字段 | 类型 | 说明 |
|---|---|---|
| `item_category` | int16 | 商品所属类目ID |

## 3. 商品维度特征（10列）

数据来源：`dim_item`（步骤二），按 item_id 关联，列名前缀 `item_`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `item_pv_count` | int32 | 该商品的总浏览次数（pv） |
| `item_view_user_count` | int32 | 浏览过该商品的去重用户数（uv） |
| `item_fav_count` | int32 | 该商品的总收藏次数 |
| `item_cart_count` | int32 | 该商品的总加购次数 |
| `item_buy_count` | int32 | 该商品的总购买次数 |
| `item_buy_user_count` | int32 | 购买过该商品的去重用户数 |
| `item_pv_to_buy_rate` | float32 | 浏览→购买转化率 = 浏览且购买的用户数 ÷ 浏览过的用户数 |
| `item_cart_to_buy_rate` | float32 | 加购→购买转化率 = 加购且购买的用户数 ÷ 加购过的用户数 |
| `item_repurchase_user_count` | int32 | 复购用户数（买过该商品≥2次的用户数） |

## 4. 类目维度特征（10列）

数据来源：`dim_category`（步骤二），按 item_category 关联，列名前缀 `cat_`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `cat_item_count` | int32 | 该类目下商品总数 |
| `cat_buy_item_count` | int32 | 该类目下有购买的商品数 |
| `cat_pv_count` | int32 | 该类目总浏览次数 |
| `cat_view_user_count` | int32 | 该类目浏览独立用户数 |
| `cat_fav_count` | int32 | 该类目总收藏次数 |
| `cat_cart_count` | int32 | 该类目总加购次数 |
| `cat_buy_count` | int32 | 该类目总购买次数 |
| `cat_buy_user_count` | int32 | 该类目购买独立用户数 |
| `cat_pv_to_buy_rate` | float32 | 类目浏览→购买转化率 = 被浏览且被买的商品数 ÷ 被浏览过的商品数 |
| `cat_buy_item_pct` | float32 | 类目下有购买商品占比 = buy_item_count / item_count |

## 5. 用户维度特征（15列）

数据来源：`dim_user`（步骤二），按 user_id 关联。行为量字段加 `user_` 前缀与商品维度区分。

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_pv_count` | int32 | 用户总浏览次数 |
| `user_fav_count` | int32 | 用户总收藏次数 |
| `user_cart_count` | int32 | 用户总加购次数 |
| `user_buy_count` | int32 | 用户总购买次数 |
| `active_days` | int32 | 用户在 31 天里有行为的天数（1-31） |
| `first_active_time` | datetime64 | 用户首次行为时间 |
| `last_active_time` | datetime64 | 用户末次行为时间 |
| `day_pct` | float32 | 白天（6-18点）行为占比 |
| `evening_pct` | float32 | 晚间（18-24点）行为占比 |
| `night_pct` | float32 | 深夜（0-6点）行为占比 |
| `buy_item_count` | int32 | 用户购买的商品种类数（去重） |
| `buy_conversion_rate` | float32 | 用户浏览→购买转化率 = 浏览且购买的商品数 ÷ 浏览过的商品数 |
| `fav_to_buy_rate` | float32 | 用户收藏→购买转化率 = 收藏且购买的商品数 ÷ 收藏过的商品数 |
| `cart_to_buy_rate` | float32 | 用户加购→购买转化率 = 加购且购买的商品数 ÷ 加购过的商品数 |
| `repurchase_item_count` | int32 | 用户复购商品种类数（买过≥2次的商品种数） |
| `is_power_user` | bool | 是否重度买家（Part1 标记） |

## 6. 生命周期特征（4列）

数据来源：从 Part1 原始明细数据计算。

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_streak_days` | int32 | 用户最长连续活跃天数（活跃定义：当天有任意行为） |
| `item_decay_slope` | float32 | 商品热度趋势斜率。对每日综合热度（pv×1+fav×2+cart×3+buy×5）做线性回归。门槛：有互动天数≥3（线性回归至少需3个点），不满足为 NaN |
| `user_avg_interval_hours` | float32 | 用户平均行为间隔（小时）。同一小时窗口合并后算相邻窗口时间差平均。只出现在 1 个窗口的用户为 NaN |
| `buy_path_type` | int8 | (用户,商品)对的购买路径分类。0=未购买该商品；1=直接购买（对该商品无收藏无加购）；2=有收藏无加购；3=有加购无收藏；4=收藏+加购都有 |

## 7. 业务导向特征（4列）

数据来源：从 dim_user + 原始明细计算。

| 字段 | 类型 | 说明 |
|---|---|---|
| `rfm_r_score` | int8 | RFM 的 R 值得分（1-5）。R=最后一次购买距 12-18 的天数分档：≤7天→5；≤14→4；≤21→3；≤28→2；>28→1。未购买用户为 1 |
| `rfm_f_score` | int8 | RFM 的 F 值得分（1-5）。F=购买次数分档：1→1；2-4→2；5-9→3；10-19→4；≥20→5。未购买用户为 1 |
| `rfm_m_score` | int8 | RFM 的 M 值得分（1-5）。M=购买商品种类数分档：1→1；2-3→2；4-6→3；7-9→4；≥10→5。未购买用户为 1 |
| `user_category_pref_score` | float32 | 用户对所购类目的偏好强度 = 该类目购买次数 / 用户总购买次数 |

## 8. 隐式语义特征（1列）

数据来源：从 Part1 原始明细数据用 TruncatedSVD 分解计算。

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_item_svd_score` | float32 | 用户对该商品的隐向量匹配分（点积）。SVD 维度=10，行为矩阵元素为综合热度（pv×1+fav×2+cart×3+buy×5） |

### 未实现特征说明

大纲步骤三要求"通过 SVD 矩阵分解提取用户-商品交互隐向量，计算商品共现相似度与关联规则特征"。
其中 SVD 矩阵分解已实现（上表 `user_item_svd_score`），**商品共现相似度与关联规则特征未实现**，原因如下：

1. **计算量级超出硬件能力**：全量 2,876,947 件商品的两两共现矩阵为 287 万 × 287 万，稠密存储需约 30,834 GB，无法在单机上计算。
2. **数据极度稀疏，不具备统计意义**：80.0% 的商品仅被 1 个用户交互过，90.4% 仅被 1-2 个用户交互过。绝大多数商品之间不存在共现关系，算出的相似度缺乏样本支撑。
3. **特征覆盖率过低**：即便将门槛提高到"被 10 个以上用户交互"（仅 39,520 件商品，占总数 1.4%），宽表中仍有 83.8% 的 (用户,商品) 对无法获得该特征值，缺失率过高，对模型帮助有限。

## 附录：数据规模

| 指标 | 值 |
|---|---|
| 主键范围 | 全量 (用户,商品) 对——所有有过任意行为（浏览/收藏/加购/购买）的对 |
| 总行数 | 4,686,904 |
| 总列数 | 47 |
| 文件大小 | ~207 MB |
| Parquet 路径 | `output/feature_wide_table.parquet` |

## 附录：缺失值说明

| 字段 | 缺失原因 |
|---|---|
| `item_decay_slope` | 商品有互动天数<3（线性回归无趋势意义），约88%商品不满足门槛 |
| `user_avg_interval_hours` | 用户只出现在 1 个小时窗口，无间隔可算 |
| `user_category_pref_score` | 未购买过任何商品的用户无类目偏好记录 |
| `user_item_svd_score` | 用户或商品在 SVD 训练矩阵中无记录时无法计算 |
