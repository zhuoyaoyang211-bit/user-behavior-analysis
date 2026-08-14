import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# 基于脚本文件位置构建数据路径，无论从哪个目录运行都能正确找到数据
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "output", "cleaned_data.parquet")

# ========== 页面配置 ==========
st.set_page_config(page_title="用户行为分析看板", layout="wide")
st.title("📊 看板⼀：特征⼯程·⽤户⾏为特征交互分析看板")

# ========== 加载数据 ==========
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.dayofweek  # 0=周一, 6=周日
    df['weekday_name'] = df['time'].dt.day_name()
    return df


df = load_data()

# ========== 侧边栏筛选器 ==========
st.sidebar.header("⚙️ 全局筛选")

min_date = df['time'].min().date()
max_date = df['time'].max().date()
date_range = st.sidebar.date_input(
    "📅 时间范围",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

behavior_options = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}
selected_behaviors = st.sidebar.multiselect(
    "🎯 行为类型",
    options=list(behavior_options.keys()),
    format_func=lambda x: behavior_options[x],
    default=list(behavior_options.keys())
)

# ========== 应用筛选 ==========
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[
        (df['time'].dt.date >= start_date) &
        (df['time'].dt.date <= end_date) &
        (df['behavior_type'].isin(selected_behaviors))
        ]
else:
    df_filtered = df[df['behavior_type'].isin(selected_behaviors)]

st.sidebar.markdown("---")
st.sidebar.info(f"当前数据：{len(df_filtered):,} 行")

# ================================================================
# 模块一：数据集核心概览区
# ================================================================
st.header("📈 模块一：数据集核心概览")

# ---- 基础指标计算 ----
total_users = df_filtered['user_id'].nunique()
total_items = df_filtered['item_id'].nunique()
total_categories = df_filtered['item_category'].nunique()
total_records = len(df_filtered)
min_date_filtered = df_filtered['time'].min().date()
max_date_filtered = df_filtered['time'].max().date()
time_span_days = (max_date_filtered - min_date_filtered).days

# 行为深度指标
avg_actions_per_user = total_records / total_users if total_users > 0 else 0
avg_items_per_user = (
    df_filtered.groupby('user_id')['item_id'].nunique().mean()
    if total_users > 0 else 0
)

# 目标关联指标
pv_count = len(df_filtered[df_filtered['behavior_type'] == 1])
buy_count = len(df_filtered[df_filtered['behavior_type'] == 4])
buy_users = df_filtered[df_filtered['behavior_type'] == 4]['user_id'].nunique()

buyer_ratio = buy_users / total_users if total_users > 0 else 0
cvr = buy_count / pv_count if pv_count > 0 else 0
avg_purchase = buy_count / total_users if total_users > 0 else 0

pos = buy_count
neg = total_records - buy_count
if pos > 0:
    pos_neg_ratio = f"1:{round(neg / pos, 1)}"
else:
    pos_neg_ratio = "N/A"

# ---- 基础规模 ----
st.subheader("📋 基础规模")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("👤 用户总数", f"{total_users:,}")
with col2:
    st.metric("📦 商品总数", f"{total_items:,}")
with col3:
    st.metric("🏷️ 类目总数", f"{total_categories:,}")
with col4:
    st.metric("📊 行为总记录数", f"{total_records:,}")
with col5:
    st.metric("📅 数据时间跨度", f"{time_span_days} 天")

# ---- 行为深度 ----
st.subheader("🔍 行为深度")
col6, col7 = st.columns(2)
with col6:
    st.metric("🖱️ 人均行为次数", f"{avg_actions_per_user:.2f}")
with col7:
    st.metric("🛍️ 人均交互商品数", f"{avg_items_per_user:.1f}")

# ---- 目标关联 ----
st.subheader("🎯 目标关联")
col10, col11, col12, col13 = st.columns(4)
with col10:
    st.metric("🛒 有购买行为用户占比", f"{buyer_ratio:.2%}")
with col11:
    st.metric("🔄 全链路整体购买转化率", f"{cvr:.2%}")
with col12:
    st.metric("⚖️ 天然正负样本比例", pos_neg_ratio)
with col13:
    st.metric("📈 人均购买次数", f"{avg_purchase:.3f}")

st.markdown("---")

# ================================================================
# 模块二：用户行为时序与分布区
# ================================================================
st.header("⏰ 模块二：用户行为时序与分布")

# ---- 2.1 用户活跃时段热力图 ----
st.subheader("🔥 用户活跃时段热力图")

# 行为类型切换控件
available_behaviors = [b for b in selected_behaviors if b in behavior_options]
if available_behaviors:
    selected_behavior = st.selectbox(
        "选择行为类型查看时段分布",
        options=available_behaviors,
        format_func=lambda x: behavior_options[x]
    )

    # 筛选数据
    heatmap_data = df_filtered[df_filtered['behavior_type'] == selected_behavior]

    if not heatmap_data.empty:
        # 按星期和小时聚合
        heatmap_agg = heatmap_data.groupby(['weekday', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_agg.pivot(index='weekday', columns='hour', values='count').fillna(0)

        # 把索引变成周几的名称
        weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        heatmap_pivot.index = [weekday_map[i] for i in heatmap_pivot.index]

        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="小时", y="星期", color="交互次数"),
            title=f"{behavior_options[selected_behavior]}行为时段分布",
            color_continuous_scale='Blues',
            aspect="auto"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 补充：时段转化率热力图（只展示有浏览和购买行为时）
        if 1 in selected_behaviors and 4 in selected_behaviors:
            st.caption("📊 时段转化率热力图（浏览→购买）")

            # 分别计算浏览和购买
            pv_data = df_filtered[df_filtered['behavior_type'] == 1]
            buy_data = df_filtered[df_filtered['behavior_type'] == 4]

            pv_agg = pv_data.groupby(['weekday', 'hour']).size().reset_index(name='pv_count')
            buy_agg = buy_data.groupby(['weekday', 'hour']).size().reset_index(name='buy_count')

            # 合并
            cvr_data = pd.merge(pv_agg, buy_agg, on=['weekday', 'hour'], how='outer').fillna(0)
            cvr_data['cvr'] = cvr_data['buy_count'] / cvr_data['pv_count']
            cvr_data['cvr'] = cvr_data['cvr'].replace([np.inf, -np.inf], 0)

            cvr_pivot = cvr_data.pivot(index='weekday', columns='hour', values='cvr').fillna(0)
            cvr_pivot.index = [weekday_map[i] for i in cvr_pivot.index]

            fig_cvr = px.imshow(
                cvr_pivot,
                labels=dict(x="小时", y="星期", color="转化率"),
                title="各时段浏览→购买转化率",
                color_continuous_scale='RdYlGn',
                aspect="auto",
                zmin=0,
                zmax=cvr_pivot.max().max() if cvr_pivot.max().max() > 0 else 1
            )
            fig_cvr.update_layout(height=400)
            st.plotly_chart(fig_cvr, use_container_width=True)
    else:
        st.info("当前筛选条件下无数据")
else:
    st.info("请至少选择一种行为类型")

# ---- 2.2 用户每日行为趋势图 ----
st.subheader("📈 用户每日行为趋势图")

# 按日期和行为类型聚合
daily_trend = df_filtered.groupby(['date', 'behavior_type']).size().reset_index(name='count')

# 按日期汇总购买转化率
daily_pv = df_filtered[df_filtered['behavior_type'] == 1].groupby('date').size().reset_index(name='pv_count')
daily_buy = df_filtered[df_filtered['behavior_type'] == 4].groupby('date').size().reset_index(name='buy_count')
daily_cvr = pd.merge(daily_pv, daily_buy, on='date', how='outer').fillna(0)
daily_cvr['cvr'] = daily_cvr['buy_count'] / daily_cvr['pv_count']
daily_cvr['cvr'] = daily_cvr['cvr'].replace([np.inf, -np.inf], 0)

# 绘制趋势图
fig_trend = go.Figure()

# 行为类型颜色映射
color_map = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

for behavior_id in selected_behaviors:
    behavior_data = daily_trend[daily_trend['behavior_type'] == behavior_id]
    if not behavior_data.empty:
        fig_trend.add_trace(go.Scatter(
            x=behavior_data['date'],
            y=behavior_data['count'],
            mode='lines+markers',
            name=behavior_options[behavior_id],
            line=dict(color=color_map.get(behavior_id, '#888'), width=2),
            marker=dict(size=3)
        ))

# 添加转化率曲线（次坐标轴）
if 4 in selected_behaviors and 1 in selected_behaviors:
    fig_trend.add_trace(go.Scatter(
        x=daily_cvr['date'],
        y=daily_cvr['cvr'],
        mode='lines+markers',
        name='转化率 (浏览→购买)',
        line=dict(color='red', dash='dash', width=2.5),
        marker=dict(size=4),
        yaxis='y2'
    ))

# 添加周末背景色块（周六、周日）
all_dates = sorted(set(daily_trend['date'].unique()) | set(daily_cvr['date'].unique()))
for d in all_dates:
    d_dt = pd.to_datetime(d)
    if d_dt.weekday() >= 5:  # 5=周六, 6=周日
        fig_trend.add_vrect(
            x0=d,
            x1=d_dt + pd.Timedelta(days=1),
            fillcolor="#E8E8E8",
            opacity=0.35,
            line_width=0,
            layer="below"
        )

fig_trend.update_layout(
    title="每日行为趋势与转化率（灰色背景为周末）",
    xaxis=dict(title="日期"),
    yaxis=dict(title="行为次数"),
    yaxis2=dict(
        title="转化率",
        overlaying='y',
        side='right',
        tickformat='.2%'
    ),
    height=450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---- 2.3 用户活跃度分布图 ----
st.subheader("📊 用户活跃度分布")

# 计算每个用户的总行为次数
user_activity = df_filtered.groupby('user_id').size().reset_index(name='total_actions')

# 统计每个用户的购买次数
user_buy = df_filtered[df_filtered['behavior_type'] == 4].groupby('user_id').size().reset_index(name='buy_count')
user_activity = pd.merge(user_activity, user_buy, on='user_id', how='left').fillna(0)
user_activity['buy_ratio'] = user_activity['buy_count'] / user_activity['total_actions']

# 使用对数间距的区间，适配长尾分布
max_actions = int(user_activity['total_actions'].max()) if not user_activity.empty else 1
log_bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
bins = [b for b in log_bins if b < max_actions * 1.2] + [max(max_actions * 2, 2)]
bins = sorted(list(set(bins)))

# 生成区间标签与几何中心（用于对数坐标定位）
labels = []
x_positions = []
for low, high in zip(bins[:-1], bins[1:]):
    if high == bins[-1]:
        labels.append(f"{int(low)}+")
    else:
        labels.append(f"{int(low)}-{int(high - 1)}")
    # 几何中心：在对数坐标上更居中
    x_positions.append(np.sqrt(max(low, 1) * high))

user_activity['activity_bin'] = pd.cut(
    user_activity['total_actions'],
    bins=bins,
    labels=labels,
    right=False
)

# 统计各区间用户数与平均转化率
activity_dist = user_activity.groupby('activity_bin', observed=True).size().reset_index(name='user_count')
activity_cvr = user_activity.groupby('activity_bin', observed=True)['buy_ratio'].mean().reset_index(
    name='avg_buy_ratio')
activity_plot = pd.merge(activity_dist, activity_cvr, on='activity_bin')
activity_plot['x_pos'] = activity_plot['activity_bin'].map(dict(zip(labels, x_positions)))

# 过滤掉空区间（保持图表简洁）
activity_plot = activity_plot[activity_plot['user_count'] > 0].sort_values('x_pos')

# 绘制双轴图（横轴为对数坐标）
fig_activity = go.Figure()

fig_activity.add_trace(go.Bar(
    x=activity_plot['x_pos'],
    y=activity_plot['user_count'],
    name='用户数',
    marker_color='#1f77b4',
    text=activity_plot['activity_bin'],
    hovertemplate='交互次数: %{text}<br>用户数: %{y:,}<extra></extra>',
    yaxis='y'
))

fig_activity.add_trace(go.Scatter(
    x=activity_plot['x_pos'],
    y=activity_plot['avg_buy_ratio'],
    mode='lines+markers',
    name='平均购买转化率',
    marker_color='red',
    line=dict(color='red', width=3),
    yaxis='y2'
))

fig_activity.update_layout(
    title="用户活跃度分布与转化率关系（横轴对数坐标）",
    xaxis=dict(
        title="用户总交互次数（对数坐标）",
        type='log',
        tickvals=activity_plot['x_pos'],
        ticktext=activity_plot['activity_bin']
    ),
    yaxis=dict(title="用户数"),
    yaxis2=dict(
        title="平均购买转化率",
        overlaying='y',
        side='right',
        tickformat='.2%'
    ),
    height=450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
st.plotly_chart(fig_activity, use_container_width=True)

# ---- 补充：用户购买次数分布 ----
st.subheader("📊 用户购买次数分布")

# 统计每个用户的购买次数
user_purchase_count = df_filtered[df_filtered['behavior_type'] == 4].groupby('user_id').size().reset_index(
    name='purchase_count')
# 统计不同购买次数的用户数
purchase_dist = user_purchase_count.groupby('purchase_count').size().reset_index(name='user_count')
# 只展示购买次数≤20的（长尾部分）
purchase_dist_filtered = purchase_dist[purchase_dist['purchase_count'] <= 20]

if not purchase_dist_filtered.empty:
    fig_purchase = px.bar(
        purchase_dist_filtered,
        x='purchase_count',
        y='user_count',
        title="用户购买次数分布（复购用户占比结构）",
        labels={'purchase_count': '购买次数', 'user_count': '用户数'},
        color_discrete_sequence=['#d62728']
    )
    fig_purchase.update_layout(height=400)
    st.plotly_chart(fig_purchase, use_container_width=True)

    # 展示复购占比统计
    total_buyers = user_purchase_count['user_id'].nunique()
    repeat_buyers = len(user_purchase_count[user_purchase_count['purchase_count'] >= 2])
    repeat_ratio = repeat_buyers / total_buyers if total_buyers > 0 else 0
    st.caption(f"📌 有购买行为的用户共 {total_buyers:,} 人，其中复购用户（购买≥2次）占 {repeat_ratio:.2%}")
else:
    st.info("当前筛选条件下无购买数据")

st.markdown("---")

# ================================================================
# 模块三：商品与类目分析区
# ================================================================
st.header("🛍️ 模块三：商品与类目分析区")

# ---- 公共聚合：受侧边栏筛选影响 ----
with st.spinner("类目 / 商品维度聚合计算中..."):
    # 类目聚合（9K 组别，lambda 方式即可）
    cat_agg = df_filtered.groupby('item_category').agg(
        pv_count=('behavior_type', lambda x: (x == 1).sum()),
        fav_count=('behavior_type', lambda x: (x == 2).sum()),
        cart_count=('behavior_type', lambda x: (x == 3).sum()),
        buy_count=('behavior_type', lambda x: (x == 4).sum()),
    ).reset_index()
    cat_agg['interaction'] = (
        cat_agg['pv_count'] + cat_agg['fav_count']
        + cat_agg['cart_count'] + cat_agg['buy_count']
    )
    cat_agg['pv_to_buy_rate'] = np.where(
        cat_agg['pv_count'] > 0,
        cat_agg['buy_count'] / cat_agg['pv_count'],
        0
    )

    # 商品聚合（百万组别，采用预筛选 + 映射方式避免 OOM）
    # item_category 映射取自完整 df，与过滤无关
    item_meta = df[['item_id', 'item_category']].drop_duplicates(subset=['item_id'])

    pv_by_item = df_filtered[df_filtered['behavior_type'] == 1].groupby('item_id').size()
    fav_by_item = df_filtered[df_filtered['behavior_type'] == 2].groupby('item_id').size()
    cart_by_item = df_filtered[df_filtered['behavior_type'] == 3].groupby('item_id').size()
    buy_by_item = df_filtered[df_filtered['behavior_type'] == 4].groupby('item_id').size()

    item_agg = item_meta.copy()
    item_agg['pv_count'] = item_agg['item_id'].map(pv_by_item).fillna(0).astype('int64')
    item_agg['fav_count'] = item_agg['item_id'].map(fav_by_item).fillna(0).astype('int64')
    item_agg['cart_count'] = item_agg['item_id'].map(cart_by_item).fillna(0).astype('int64')
    item_agg['buy_count'] = item_agg['item_id'].map(buy_by_item).fillna(0).astype('int64')
    item_agg['interaction'] = (
        item_agg['pv_count'] + item_agg['fav_count']
        + item_agg['cart_count'] + item_agg['buy_count']
    )
    item_agg['pv_to_buy_rate'] = np.where(
        item_agg['pv_count'] > 0,
        item_agg['buy_count'] / item_agg['pv_count'],
        0
    )

# === 3.1 热门类目双维度排名 ===
st.subheader("🏆 热门类目双维度排名")

cat_sort = st.radio(
    "排序方式",
    ["按交互量", "按购买转化率"],
    horizontal=True,
    key="cat_sort_metric"
)
top_n_cat = 15
# 按转化率排序时排除交互量太小的类目，避免噪声
eligible_cat = cat_agg if cat_sort == "按交互量" else cat_agg[cat_agg['interaction'] >= 1000]
if cat_sort == "按交互量":
    cat_top = eligible_cat.nlargest(top_n_cat, 'interaction').sort_values('interaction', ascending=True)
    x_values = cat_top['interaction']
    x_title = "交互量"
else:
    cat_top = eligible_cat.nlargest(top_n_cat, 'pv_to_buy_rate').sort_values('pv_to_buy_rate', ascending=True)
    x_values = cat_top['pv_to_buy_rate']
    x_title = "购买转化率"

cat_labels = [f"类目 {c}" for c in cat_top['item_category']]
customdata_cat = list(zip(
    cat_top['pv_count'], cat_top['fav_count'],
    cat_top['cart_count'], cat_top['buy_count'],
    cat_top['pv_to_buy_rate']
))

fig_cat = go.Figure()
fig_cat.add_trace(go.Bar(
    y=cat_labels,
    x=x_values,
    orientation='h',
    marker=dict(
        color=cat_top['pv_to_buy_rate'],
        colorscale='Blues',
        cmin=0,
        cmax=max(cat_top['pv_to_buy_rate'].max(), 1e-6),
        colorbar=dict(title="转化率", tickformat='.2%'),
        showscale=True
    ),
    text=[f"PV {r.pv_count:,} | 转化 {r.pv_to_buy_rate:.2%}" for r in cat_top.itertuples()],
    textposition='outside',
    customdata=customdata_cat,
    hovertemplate=(
        '<b>%{y}</b><br>'
        '交互量: %{x:,}<br>'
        'PV: %{customdata[0]:,}<br>'
        '收藏: %{customdata[1]:,}<br>'
        '加购: %{customdata[2]:,}<br>'
        '购买: %{customdata[3]:,}<br>'
        '转化率: %{customdata[4]:.2%}<extra></extra>'
    )
))
fig_cat.update_layout(
    title=f"Top {top_n_cat} 热门类目（按{cat_sort.replace('按', '')}排序）",
    xaxis=dict(title=x_title),
    yaxis=dict(title=""),
    height=max(450, 32 * len(cat_top) + 80),
    margin=dict(l=120)
)
st.plotly_chart(fig_cat, use_container_width=True)

# === 3.2 热门商品双维度排名 ===
st.subheader("🏅 热门商品双维度排名（Top 20）")

item_sort = st.radio(
    "排序方式",
    ["按交互量", "按购买转化率"],
    horizontal=True,
    key="item_sort_metric"
)
top_n_item = 20
eligible_item = item_agg if item_sort == "按交互量" else item_agg[item_agg['pv_count'] >= 100]
if item_sort == "按交互量":
    item_top = eligible_item.nlargest(top_n_item, 'interaction').sort_values('interaction', ascending=True)
    x_values_item = item_top['interaction']
    x_title_item = "交互量"
else:
    item_top = eligible_item.nlargest(top_n_item, 'pv_to_buy_rate').sort_values('pv_to_buy_rate', ascending=True)
    x_values_item = item_top['pv_to_buy_rate']
    x_title_item = "购买转化率"

item_labels = [
    f"商品 {i} · 类目 {c}" for i, c in zip(item_top['item_id'], item_top['item_category'])
]
customdata_item = list(zip(
    item_top['item_category'], item_top['pv_count'],
    item_top['buy_count'], item_top['pv_to_buy_rate']
))

fig_item = go.Figure()
fig_item.add_trace(go.Bar(
    y=item_labels,
    x=x_values_item,
    orientation='h',
    marker=dict(color='#2ca02c'),
    text=[
        f"PV {r.pv_count:,} | 转化 {r.pv_to_buy_rate:.2%}"
        for r in item_top.itertuples()
    ],
    textposition='outside',
    customdata=customdata_item,
    hovertemplate=(
        '<b>%{y}</b><br>'
        '所属类目: %{customdata[0]}<br>'
        '浏览量: %{customdata[1]:,}<br>'
        '购买量: %{customdata[2]:,}<br>'
        '转化率: %{customdata[3]:.2%}<extra></extra>'
    )
))
fig_item.update_layout(
    title=f"Top {top_n_item} 热门商品（按{item_sort.replace('按', '')}排序）",
    xaxis=dict(title=x_title_item),
    yaxis=dict(title=""),
    height=max(550, 30 * len(item_top) + 80),
    margin=dict(l=180)
)
st.plotly_chart(fig_item, use_container_width=True)

# === 3.3 类目长尾分布图 ===
st.subheader("📉 类目长尾分布图")

# 类目按交互量降序排序
cat_sorted = cat_agg.sort_values('interaction', ascending=False).reset_index(drop=True)
total_interaction = cat_sorted['interaction'].sum()
total_buy = cat_sorted['buy_count'].sum()
if total_interaction > 0:
    cat_sorted['cum_interaction_pct'] = cat_sorted['interaction'].cumsum() / total_interaction * 100
else:
    cat_sorted['cum_interaction_pct'] = 0
if total_buy > 0:
    cat_sorted['cum_buy_pct'] = cat_sorted['buy_count'].cumsum() / total_buy * 100
else:
    cat_sorted['cum_buy_pct'] = 0

fig_long = go.Figure()
fig_long.add_trace(go.Scatter(
    x=list(range(1, len(cat_sorted) + 1)),
    y=cat_sorted['cum_interaction_pct'],
    mode='lines',
    name='累计交互量占比',
    line=dict(color='#1f77b4', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(31, 119, 180, 0.1)'
))
fig_long.add_trace(go.Scatter(
    x=list(range(1, len(cat_sorted) + 1)),
    y=cat_sorted['cum_buy_pct'],
    mode='lines',
    name='累计购买占比',
    line=dict(color='#d62728', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(214, 39, 40, 0.08)'
))
# 80% 参考线
fig_long.add_hline(y=80, line_dash='dash', line_color='gray',
                   annotation_text='80%', annotation_position='right')
# 20% 类目参考线
top_20pct_mark = int(len(cat_sorted) * 0.2)
if top_20pct_mark > 0:
    fig_long.add_vline(x=top_20pct_mark, line_dash='dash', line_color='lightgray',
                       annotation_text='Top 20%', annotation_position='top')

fig_long.update_layout(
    title="类目长尾分布（验证二八效应：少数类目贡献大部分流量与成交）",
    xaxis=dict(title="类目数量（按交互量降序）"),
    yaxis=dict(title="累计占比 (%)", range=[0, 105]),
    height=450,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
st.plotly_chart(fig_long, use_container_width=True)

# 计算并展示帕累托指标
if len(cat_sorted) > 0 and total_interaction > 0:
    top_20pct_n = max(1, int(len(cat_sorted) * 0.2))
    share_traffic_20 = cat_sorted.iloc[:top_20pct_n]['interaction'].sum() / total_interaction
    share_buy_20 = (
        cat_sorted.iloc[:top_20pct_n]['buy_count'].sum() / total_buy
        if total_buy > 0 else 0
    )
    st.caption(
        f"📌 头部 {top_20pct_n} 个类目（前 20%）贡献了 "
        f"{share_traffic_20:.2%} 的交互量 和 {share_buy_20:.2%} 的购买量。"
        f"长尾由 {len(cat_sorted) - top_20pct_n} 个类目构成。"
    )

    

# === 3.4 商品曝光-转化四象限图 ===
st.subheader("🎯 商品曝光-转化四象限图")

# 取曝光量 Top 的商品用于分析（避免极小样本干扰）
top_n_quad = st.slider(
    "选取商品范围（按曝光量取头部 N 个商品）",
    min_value=500, max_value=10000,
    value=3000, step=500,
    key="quad_top_n"
)
items_quad = item_agg[item_agg['pv_count'] > 0].nlargest(top_n_quad, 'pv_count').copy()

# 用中位数作为象限分界（中位数对长尾分布更具稳健性）
pv_threshold = items_quad['pv_count'].median()
rate_threshold = items_quad['pv_to_buy_rate'].median()

# 向量化四象限分类
high_exp = items_quad['pv_count'] >= pv_threshold
high_cvr = items_quad['pv_to_buy_rate'] >= rate_threshold
items_quad['quadrant'] = np.where(
    high_exp & high_cvr, '明星商品',
    np.where(high_exp & ~high_cvr, '流量商品',
    np.where(~high_exp & high_cvr, '潜力商品', '长尾商品'))
)

quadrant_order = ['明星商品', '流量商品', '潜力商品', '长尾商品']
quadrant_colors = {
    '明星商品': '#d62728',   # 红 - 高曝光高转化
    '潜力商品': '#2ca02c',   # 绿 - 低曝光高转化
    '流量商品': '#ff7f0e',   # 橙 - 高曝光低转化
    '长尾商品': '#9aa0a6'    # 灰 - 低曝光低转化
}

fig_quad = go.Figure()
for q in quadrant_order:
    subset = items_quad[items_quad['quadrant'] == q]
    if subset.empty:
        continue
    fig_quad.add_trace(go.Scattergl(
        x=subset['pv_count'],
        y=subset['pv_to_buy_rate'],
        mode='markers',
        name=f"{q} ({len(subset)})",
        marker=dict(
            size=np.clip(np.sqrt(subset['pv_count']) * 1.2, 4, 30),
            color=quadrant_colors[q],
            opacity=0.55,
            line=dict(width=0)
        ),
        hovertemplate=(
            f'<b>{q}</b><br>'
            '商品ID: %{customdata[0]}<br>'
            '所属类目: %{customdata[1]}<br>'
            '浏览量: %{x:,}<br>'
            '转化率: %{y:.2%}<extra></extra>'
        ),
        customdata=list(zip(subset['item_id'], subset['item_category']))
    ))

# 象限分界线
fig_quad.add_vline(x=pv_threshold, line_dash='dash', line_color='gray', opacity=0.6)
fig_quad.add_hline(y=rate_threshold, line_dash='dash', line_color='gray', opacity=0.6)

# 象限标签
x_mid = items_quad['pv_count'].quantile(0.75)
y_mid = items_quad['pv_to_buy_rate'].quantile(0.75)
annotations = [
    dict(x=items_quad['pv_count'].quantile(0.9), y=items_quad['pv_to_buy_rate'].quantile(0.9),
         text='明星区<br>(高曝光·高转化)', showarrow=False,
         font=dict(color=quadrant_colors['明星商品'], size=11), opacity=0.7),
    dict(x=items_quad['pv_count'].quantile(0.9), y=items_quad['pv_to_buy_rate'].quantile(0.1),
         text='流量区<br>(高曝光·低转化)', showarrow=False,
         font=dict(color=quadrant_colors['流量商品'], size=11), opacity=0.7),
    dict(x=items_quad['pv_count'].quantile(0.1), y=items_quad['pv_to_buy_rate'].quantile(0.9),
         text='潜力区<br>(低曝光·高转化)', showarrow=False,
         font=dict(color=quadrant_colors['潜力商品'], size=11), opacity=0.7),
    dict(x=items_quad['pv_count'].quantile(0.1), y=items_quad['pv_to_buy_rate'].quantile(0.1),
         text='长尾区<br>(低曝光·低转化)', showarrow=False,
         font=dict(color=quadrant_colors['长尾商品'], size=11), opacity=0.7),
]
fig_quad.update_layout(
    title=f"商品曝光-转化四象限（Top{top_n_quad}商品，分界线=中位数）",
    xaxis=dict(title="商品曝光量（对数坐标）", type='log'),
    yaxis=dict(title="购买转化率", tickformat='.2%'),
    height=560,
    annotations=annotations,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)
st.plotly_chart(fig_quad, use_container_width=True)

# 象限统计表
quad_stats = items_quad.groupby('quadrant').agg(
    商品数=('item_id', 'count'),
    总曝光=('pv_count', 'sum'),
    总购买=('buy_count', 'sum'),
    平均转化率=('pv_to_buy_rate', 'mean')
).reindex(quadrant_order).reset_index()
quad_stats['占商品数比'] = quad_stats['商品数'] / quad_stats['商品数'].sum()
quad_stats['平均转化率'] = quad_stats['平均转化率'].map(lambda x: f"{x:.2%}")

st.markdown("**📊 四象限分布统计**")
st.dataframe(quad_stats, use_container_width=True, hide_index=True)


# ================================================================
# 模块四：转化链路与特征关联区（特征工程核心模块 · 看板 2）
# ================================================================
st.markdown("---")
st.header("🎯 模块四：转化链路与特征关联")
st.caption("💡 核心作用：直接验证特征与目标变量的相关性，指导特征体系设计")

# ---- 4.0 局部筛选区（看板 2 全局交互控件）----
with st.container():
    st.markdown("### 🎛️ 局部筛选（仅本看板生效，不影响模块 1-3）")

    # 加载 dim_user 用于用户分层（10K 行，常驻加载成本可忽略）
    BASE_DIR_M4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DIM_USER_PATH = os.path.join(BASE_DIR_M4, "output", "dim_user.parquet")
    dim_user_full = pd.read_parquet(DIM_USER_PATH)

    # 全量数据用于类目多选和时间范围基础值（df 已由 load_data cache 准备）
    df_all = df

    # 商品类目多选：基于 Top 50 热门，避免 8916 个 multiselect 卡死
    top_categories = (
        df_all['item_category'].value_counts().head(50).index.tolist()
    )
    top_categories_sorted = sorted(top_categories)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        st.markdown("**📅 时间范围**")
        min_d = df_all['time'].min().date()
        max_d = df_all['time'].max().date()
        local_date_range = st.date_input(
            "选择时间范围",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key='m4_date_range'
        )
    with ctrl_col2:
        st.markdown("**👥 用户分层**")
        pv_q25 = float(dim_user_full['pv_count'].quantile(0.25))
        pv_q75 = float(dim_user_full['pv_count'].quantile(0.75))
        power_n = int(dim_user_full['is_power_user'].sum())
        segment_options = {
            'all': f'全部用户 ({len(dim_user_full):,})',
            'low': f'低活跃 PV<{pv_q25:.0f}',
            'mid': f'中活跃 {pv_q25:.0f}≤PV<{pv_q75:.0f}',
            'high': f'高活跃 PV≥{pv_q75:.0f}',
            'power': f'超级用户 is_power_user=True ({power_n})',
        }
        local_segment = st.selectbox(
            "选择用户分层（基于 dim_user 静态画像）",
            options=list(segment_options.keys()),
            format_func=lambda x: segment_options[x],
            index=0,
            key='m4_segment'
        )
    with ctrl_col3:
        st.markdown("**🏷️ 商品类目（Top 50）**")
        local_categories = st.multiselect(
            "选择热门类目（默认全选）",
            options=top_categories_sorted,
            default=top_categories_sorted,
            format_func=lambda x: f"类目 {x}",
            key='m4_categories'
        )

    # ---- 派生 df_m4：受本看板筛选影响 ----
    if len(local_date_range) == 2:
        local_start, local_end = local_date_range
    else:
        local_start, local_end = min_d, max_d

    if local_segment == 'low':
        seg_users = set(
            dim_user_full[dim_user_full['pv_count'] < pv_q25].index.tolist()
        )
    elif local_segment == 'mid':
        seg_users = set(
            dim_user_full[
                (dim_user_full['pv_count'] >= pv_q25)
                & (dim_user_full['pv_count'] < pv_q75)
            ].index.tolist()
        )
    elif local_segment == 'high':
        seg_users = set(
            dim_user_full[dim_user_full['pv_count'] >= pv_q75].index.tolist()
        )
    elif local_segment == 'power':
        seg_users = set(
            dim_user_full[dim_user_full['is_power_user'] == True].index.tolist()
        )
    else:  # 'all'
        seg_users = set(dim_user_full.index.tolist())

    df_m4 = df_all[
        (df_all['time'].dt.date >= local_start)
        & (df_all['time'].dt.date <= local_end)
        & (df_all['user_id'].isin(seg_users))
        & (df_all['item_category'].isin(local_categories))
    ].copy()

    st.caption(
        f"📊 本看板当前筛选：**{len(seg_users):,}** 个用户 | "
        f"**{len(local_categories)}** 个类目 | "
        f"时间 **{local_start} → {local_end}** | "
        f"数据量 **{len(df_m4):,}** 行"
    )

st.markdown("")


# ---- 4.1 全链路行为转化漏斗图 ----
st.subheader("🌊 4.1 全链路行为转化漏斗图")
st.caption(
    "📌 按指定顺序（浏览→加购→收藏→购买）展示各级**触达用户数**。"
    "由于加购 / 收藏 / 购买是独立可触达行为（非必经环节），"
    "本漏斗近似展示该路径上的用户规模衰减与各环节相对上一级的转化效率。"
)

with st.spinner("计算各级触达用户数..."):
    # 强制考虑全部 4 种行为（不受其他筛选限制，专注漏斗展示）
    df_funnel_src = df_m4[df_m4['behavior_type'].isin([1, 2, 3, 4])]
    user_pivot = df_funnel_src.groupby('behavior_type')['user_id'].nunique()

    # 用户指定顺序：浏览→加购→收藏→购买
    funnel_def = [
        (1, '浏览', '#5470c6'),  # 蓝
        (3, '加购', '#f5a742'),  # 橙
        (2, '收藏', '#66c2a5'),  # 绿
        (4, '购买', '#ee6677'),  # 红
    ]
    rows = []
    prev = None
    for beh, name, color in funnel_def:
        cnt = int(user_pivot.get(beh, 0))
        if prev is None or prev == 0:
            cvr_prev, drop_prev = 1.0, 0.0
        else:
            cvr_prev = cnt / prev if prev > 0 else 0
            drop_prev = 1 - cvr_prev
        rows.append({
            '阶段': name,
            '触达用户数': cnt,
            '占入口比例': None,
            '相对上一级转化率': cvr_prev,
            '相对上一级流失率': drop_prev,
            '_color': color,
        })
        prev = cnt
    funnel_df = pd.DataFrame(rows)
    entry_users = int(funnel_df.iloc[0]['触达用户数'])
    funnel_df['占入口比例'] = (
        funnel_df['触达用户数'] / entry_users if entry_users > 0 else 0
    )

# 漏斗图：自定义横向条形漏斗（保留用户指定顺序，避免非严格降序导致 plotly funnel 反弹）
# 暖色阶过渡：浏览(蓝) → 加购(橙) → 收藏(绿) → 购买(红)，对应流量→成交的色彩语义
fig_funnel = go.Figure()
for _, row in funnel_df.iterrows():
    fig_funnel.add_trace(go.Bar(
        x=[row['触达用户数']],
        y=[row['阶段']],
        orientation='h',
        marker=dict(
            color=row['_color'],
            line=dict(width=2, color='white'),
        ),
        text=[f"<b>{row['阶段']}</b><br>"
              f"{row['触达用户数']:,} 用户（占入口 {row['占入口比例']:.1%}）"],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(size=13, color='white', family='Arial Black'),
        width=0.62,
        showlegend=False,
        hovertemplate=(
            f"<b>{row['阶段']}</b><br>"
            f"触达用户: {int(row['触达用户数']):,}<br>"
            f"占入口: {row['占入口比例']:.2%}<br>"
            f"相对上一级转化: {row['相对上一级转化率']:.2%}<extra></extra>"
        ),
    ))

# 连接相邻段的灰色虚线（视觉上的"漏斗连接"）
for i in range(len(funnel_df) - 1):
    cur = funnel_df.iloc[i]
    nxt = funnel_df.iloc[i + 1]
    fig_funnel.add_shape(
        type='line',
        x0=cur['触达用户数'], x1=nxt['触达用户数'],
        y0=i, y1=i + 1,
        line=dict(color='lightgray', dash='dot', width=2),
        layer='below',
    )

fig_funnel.update_layout(
    title=dict(
        text=f'全链路行为转化漏斗 · 总入口 {entry_users:,} 用户',
        font=dict(size=16),
    ),
    height=440,
    margin=dict(l=110, r=80, t=80, b=40),
    yaxis=dict(autorange='reversed', title=''),  # 浏览在最上
    xaxis=dict(
        title='触达用户数',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
    ),
    plot_bgcolor='rgba(250,250,250,0.5)',
    bargap=0.05,
)
st.plotly_chart(fig_funnel, use_container_width=True)

# 各级明细表
def _fmt_cvr(v, is_first):
    if is_first:
        return '—'
    if v > 1.0001:  # 独立触达口径会出现 >100%，显式标注
        return f'{v:.2%} ↑独立路径'
    return f'{v:.2%}'


def _fmt_drop(v, is_first):
    if is_first:
        return '—'
    if v < -0.0001:  # 下一级超过上一级时流失率为负数（独立路径），不展示
        return '↑ 独立路径'
    return f'{v:.2%}'


funnel_disp = funnel_df.copy()
funnel_disp['占入口比例'] = funnel_disp['占入口比例'].map(lambda x: f'{x:.2%}')
funnel_disp['相对上一级转化率'] = [
    _fmt_cvr(v, i == 0)
    for i, v in enumerate(funnel_disp['相对上一级转化率'])
]
funnel_disp['相对上一级流失率'] = [
    _fmt_drop(v, i == 0)
    for i, v in enumerate(funnel_disp['相对上一级流失率'])
]
funnel_disp = funnel_disp[
    ['阶段', '触达用户数', '占入口比例', '相对上一级转化率', '相对上一级流失率']
]

col_l, col_r = st.columns([3, 2])
with col_l:
    st.markdown('**📊 各级转化与流失明细**')
    st.dataframe(funnel_disp, use_container_width=True, hide_index=True)
with col_r:
    final_cvr = float(funnel_df.iloc[-1]['占入口比例'])
    worst_drop_idx = int(funnel_df['相对上一级流失率'].idxmax())
    worst_drop_row = funnel_df.iloc[worst_drop_idx]
    worst_drop_step = worst_drop_row['阶段']
    worst_drop_val = float(worst_drop_row['相对上一级流失率'])
    shape_type = '长尾型' if final_cvr < 0.05 else '正常衰减型'
    st.markdown(
        f'**🎯 核心结论**  \n\n'
        f'- 📍 **端到端转化**（浏览→购买）：**{final_cvr:.2%}**  \n'
        f'- ⚠️ **最大流失环节**：**{worst_drop_step}**，流失率 '
        f'**{worst_drop_val:.2%}**  \n'
        f'- 📐 **形态**：`{shape_type}`  \n'
        f'- 💡 **特征启示**：该环节的"用户是否流失"信号可作为负样本权重特征。'
    )

st.markdown('---')


# ---- 4.2 行为跳转概率矩阵 ----
st.subheader('🔀 4.2 行为跳转概率矩阵')
st.caption(
    '💡 计算逻辑：同一用户**相邻时间的两条行为记录**之间的转移概率。'
    '下一条不属于同一用户时记为"结束序列"。'
    '支撑 **行为序列特征工程**（GRU / Transformer 序列建模、注意力权重设计）。'
)

with st.spinner('计算跳转矩阵中（约 5-15 秒）...'):
    seq_df = df_m4[['user_id', 'behavior_type', 'time']].copy()
    seq_df = seq_df.sort_values(
        ['user_id', 'time'], kind='mergesort'
    ).reset_index(drop=True)

    seq_df['next_user'] = seq_df['user_id'].shift(-1)
    seq_df['next_beh'] = seq_df['behavior_type'].shift(-1)

    # 是否属于同一用户的相邻记录（不做 session 切分：
    # 小时级数据精度不足以做精细 session，跨天跳转也算相邻行为）
    is_same_user_next = (seq_df['next_user'] == seq_df['user_id'])

    # 0 表示"无下一条记录 / 结束序列"，作为虚拟下一步行为
    seq_df['next_eff'] = np.where(is_same_user_next, seq_df['next_beh'], 0)

    trans_agg = (
        seq_df.groupby(['behavior_type', 'next_eff'])
        .size().reset_index(name='count')
    )

# ---- 构造 4×5 矩阵（行=当前行为，列=下一步或退出）----
behavior_ids = [1, 2, 3, 4]
beh_label = {1: '浏览', 2: '收藏', 3: '加购', 4: '购买'}
next_label = {1: '浏览', 2: '收藏', 3: '加购', 4: '购买', 0: '▶ 结束序列'}

matrix = pd.DataFrame(0.0, index=behavior_ids, columns=behavior_ids + [0])
for _, r in trans_agg.iterrows():
    matrix.loc[int(r['behavior_type']), int(r['next_eff'])] = r['count']

# 行归一化为概率
row_sums = matrix.sum(axis=1).replace(0, 1)
matrix_pct = matrix.div(row_sums, axis=0) * 100

row_labels = [f'当前: {beh_label[b]}' for b in matrix.index]
col_labels = [next_label[c] for c in matrix.columns]

# 文本标注：>0.05% 显示 1 位小数，<0.05% 显示 "<0.1%"
text_display = np.where(
    matrix_pct.values > 0.05,
    np.round(matrix_pct.values, 1).astype(str) + '%',
    '<0.1%',
)

# RdYlGn 暖色高概率 / 冷色低概率（配色规范）
matrix_max = max(float(matrix_pct.values.max()), 1.0)

fig_matrix = go.Figure(data=go.Heatmap(
    z=matrix_pct.values,
    x=col_labels,
    y=row_labels,
    colorscale='RdYlGn',
    zmin=0,
    zmax=matrix_max,
    text=text_display,
    texttemplate='%{text}',
    textfont=dict(size=12, color='#333'),
    colorbar=dict(title='转移概率 %', ticksuffix='%'),
    hovertemplate=(
        '当前: %{y}<br>下一步: %{x}<br>'
        '概率: %{z:.2f}%<extra></extra>'
    ),
    xgap=2, ygap=2,
))

fig_matrix.update_layout(
    title='行为跳转概率矩阵（同行加总 = 100%）',
    xaxis_title='下一步行为',
    yaxis_title='当前行为',
    height=440,
    yaxis=dict(autorange='reversed'),  # "浏览"放最上方
    margin=dict(l=130, r=20, t=80, b=60),
)
st.plotly_chart(fig_matrix, use_container_width=True)

# ---- Top 3 关键流转对（剔除"退出"列）----
matrix_inner = matrix_pct.iloc[:, :-1]
top_pairs = matrix_inner.stack().nlargest(3)

pair_records = []
for (curr_b, next_b), pct in top_pairs.items():
    pair_records.append({
        '当前行为': beh_label[int(curr_b)],
        '下一步行为': beh_label[int(next_b)],
        '转移概率': f'{pct:.2f}%',
        '原始计数': f'{int(matrix.loc[int(curr_b), int(next_b)]):,}',
    })
st.markdown('**🎯 Top 3 关键行为流转对**')
st.dataframe(pd.DataFrame(pair_records), use_container_width=True, hide_index=True)

st.caption(
    '💡 例如若 P(浏览→加购) 显著高于 P(浏览→收藏)，'
    '在序列模型中应给予"加购"更高的注意力权重。'
)

st.markdown('---')


# ---- 4.3 核心行为深度-转化率关联图 ----
st.subheader('📈 4.3 核心行为深度-转化率关联图')
st.caption(
    '💡 横轴：浏览 / 加购 / 收藏次数的对数分箱区间；'
    '纵轴：该区间用户的 **真实购买率**（有购买行为人数 / 总人数）。'
    '验证 **行为深度特征** 对购买转化的预测能力。'
)

with st.spinner('计算各行为深度区间的真实购买率...'):
    # df_m4 上重新聚合，保持与本看板筛选一致
    valid_users_m4 = df_m4['user_id'].unique()
    user_metric = pd.DataFrame(index=valid_users_m4)

    cnt_1 = df_m4[df_m4['behavior_type'] == 1].groupby('user_id').size()
    cnt_2 = df_m4[df_m4['behavior_type'] == 2].groupby('user_id').size()
    cnt_3 = df_m4[df_m4['behavior_type'] == 3].groupby('user_id').size()
    buyers = df_m4[df_m4['behavior_type'] == 4].groupby('user_id').size()

    user_metric['count_1'] = cnt_1
    user_metric['count_2'] = cnt_2
    user_metric['count_3'] = cnt_3
    user_metric['has_buy'] = (
        user_metric.index.map(buyers).fillna(0) > 0
    ).astype(int)
    user_metric = user_metric.fillna(0)

# 三面板配置
from plotly.subplots import make_subplots

depth_specs = [
    ('count_1', '浏览深度', '#d62728'),   # 红
    ('count_3', '加购深度', '#1f77b4'),   # 蓝
    ('count_2', '收藏深度', '#2ca02c'),   # 绿
]


def _log_bins(max_v):
    base = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    bins = [b for b in base if b < max_v * 1.2]
    bins.append(max(int(max_v * 2), 2))
    return sorted(set(bins))


def _label_for_bins(bins):
    labels = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            labels.append(f'{int(bins[i])}+')
        else:
            labels.append(f'{int(bins[i])}-{int(bins[i + 1] - 1)}')
    return labels


def _aggregate_depth(series, has_buy):
    max_v = float(series.max()) if series.max() > 0 else 1
    bins = _log_bins(max_v)
    labels = _label_for_bins(bins)
    tmp = pd.DataFrame({'count': series, 'has_buy': has_buy})
    tmp['bin'] = pd.cut(tmp['count'], bins=bins, labels=labels, right=False)
    grouped = tmp.groupby('bin', observed=True).agg(
        users=('has_buy', 'count'),
        buyers=('has_buy', 'sum'),
    )
    grouped['buy_rate'] = grouped['buyers'] / grouped['users']
    return grouped[grouped['users'] > 0].reset_index()


buy_rate_overall = float(user_metric['has_buy'].mean())

fig_depth = make_subplots(
    rows=1, cols=3,
    subplot_titles=[s[1] for s in depth_specs],
    horizontal_spacing=0.08,
)

# 三个深度的曲线与峰值标注
for col_idx, (col_name, title, color) in enumerate(depth_specs, start=1):
    grouped = _aggregate_depth(
        user_metric[col_name], user_metric['has_buy']
    )
    if grouped.empty:
        fig_depth.add_annotation(
            text='当前筛选下无该行为深度数据',
            showarrow=False,
            xref=f'x{col_idx}', yref=f'y{col_idx}',
            x=0.5, y=0.5,
            row=1, col=col_idx,
        )
        continue
    x_labels = grouped['bin'].astype(str).tolist()

    fig_depth.add_trace(
        go.Scatter(
            x=x_labels,
            y=grouped['buy_rate'],
            mode='lines+markers',
            name=title,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(width=1, color='white')),
            text=[
                f'{int(u):,}用户 / {int(b):,}购买'
                for u, b in zip(grouped['users'], grouped['buyers'])
            ],
            hovertemplate=(
                f'{title}<br>'
                '区间: %{x}<br>'
                '购买率: %{y:.2%}<br>'
                '%{text}<extra></extra>'
            ),
            showlegend=False,
        ),
        row=1, col=col_idx,
    )

    # 全局购买率参考线（冷灰色 = 中性参考）
    fig_depth.add_hline(
        y=buy_rate_overall,
        line=dict(color='gray', dash='dot', width=1.5),
        annotation_text=f'全局 {buy_rate_overall:.2%}',
        annotation_position='top right',
        annotation_font_size=10,
        row=1, col=col_idx,
    )

    # 峰值点数值标注（暖色 = 正向高亮）
    peak_idx = int(grouped['buy_rate'].idxmax())
    peak_row = grouped.iloc[peak_idx]
    fig_depth.add_annotation(
        x=str(peak_row['bin']),
        y=float(peak_row['buy_rate']),
        text=f'峰 {peak_row["buy_rate"]:.2%}',
        showarrow=True,
        arrowhead=2,
        arrowcolor=color,
        ax=0, ay=-25,
        font=dict(color=color, size=10, family='Arial Black'),
        row=1, col=col_idx,
    )

fig_depth.update_layout(
    height=460,
    title_text='行为深度 × 真实购买率：横向三面板对照',
    margin=dict(l=20, r=20, t=80, b=80),
)
fig_depth.update_xaxes(tickangle=-30)
fig_depth.update_yaxes(tickformat='.2%')

# y 轴自适应：根据最大值决定
y_top = max(0.5, buy_rate_overall * 3)
fig_depth.update_yaxes(range=[0, min(1.0, y_top)])
fig_depth.update_xaxes(title_text='行为次数（对数分箱）')

st.plotly_chart(fig_depth, use_container_width=True)

# ---- 三面板结论表 ----
depth_concl = []
for col_name, title, color in depth_specs:
    grouped = _aggregate_depth(
        user_metric[col_name], user_metric['has_buy']
    )
    if grouped.empty:
        depth_concl.append({
            '深度': title, '单调性': '—', '峰值区间': '—',
            '峰值购买率': '—', '与全局对比': '—'
        })
        continue
    peak_idx = int(grouped['buy_rate'].idxmax())
    peak = grouped.iloc[peak_idx]
    xseq = list(range(len(grouped)))
    if len(xseq) >= 2:
        corr = float(np.corrcoef(xseq, grouped['buy_rate'].values)[0, 1])
        if corr > 0.3:
            mono = '↑ 单调上升（深度↑ → 转化↑）'
        elif corr < -0.3:
            mono = '↓ 单调下降（饱和或异常）'
        else:
            mono = '↔ 波动型（非简单线性）'
    else:
        mono = '单点'
    sign = '🔺 高于全局' if peak['buy_rate'] > buy_rate_overall else '🔻 低于全局'
    depth_concl.append({
        '深度': title,
        '单调性': mono,
        '峰值区间': str(peak['bin']),
        '峰值购买率': f"{peak['buy_rate']:.2%}",
        '与全局对比': sign,
    })

st.markdown(
    f'**🎯 行为深度特征权重启示**（全局购买率参考线 = **{buy_rate_overall:.2%}**）'
)
st.dataframe(pd.DataFrame(depth_concl), use_container_width=True, hide_index=True)

st.caption(
    '💡 若某深度呈"单调上升"，该深度特征在排序 / 预估模型中应给予较高权重；'
    '"波动型"则适合做非线性变换（log、分箱 Embedding 或 Tree 模型自动捕捉）。'
)