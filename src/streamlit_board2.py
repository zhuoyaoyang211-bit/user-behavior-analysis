import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json

from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# 基于脚本文件位置构建数据路径，无论从哪个目录运行都能正确找到数据
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")
EVAL_DIR = os.path.join(OUT_DIR, "step9_test_model_evaluation")
INTERP_DIR = os.path.join(OUT_DIR, "step9_test_interpretability_error_analysis")

# 评估所用分类阈值（与步骤九保持一致，来自内部调优验证集的选定阈值）
DEFAULT_THRESHOLD = 0.2

# 模型 24 个特征列（与训练 artifact 的 feature_cols 顺序一致）
FEATURE_COLS = [
    "item_pv_count", "item_fav_count", "item_cart_count", "item_buy_count",
    "item_buy_user_count", "item_pv_to_buy_rate", "item_cart_to_buy_rate",
    "item_repurchase_user_count", "cat_pv_count", "cat_view_user_count",
    "user_pv_count", "day_pct", "night_pct", "buy_conversion_rate",
    "fav_to_buy_rate", "cart_to_buy_rate", "repurchase_item_count",
    "item_decay_slope", "user_avg_interval_hours", "rfm_r_score",
    "rfm_f_score", "rfm_m_score", "user_category_pref_score", "item_category_te",
]

# 业务友好的特征名映射（讲解/汇报时使用）
FEATURE_CN = {
    "item_pv_count": "商品浏览数", "item_fav_count": "商品收藏数",
    "item_cart_count": "商品加购数", "item_buy_count": "商品购买数",
    "item_buy_user_count": "商品购买人数", "item_pv_to_buy_rate": "商品浏览→购买率",
    "item_cart_to_buy_rate": "商品加购→购买率", "item_repurchase_user_count": "商品复购人数",
    "cat_pv_count": "类目浏览数", "cat_view_user_count": "类目浏览人数",
    "user_pv_count": "用户浏览数", "day_pct": "白天行为占比",
    "night_pct": "夜间行为占比", "buy_conversion_rate": "用户购买转化率",
    "fav_to_buy_rate": "收藏→购买率", "cart_to_buy_rate": "加购→购买率",
    "repurchase_item_count": "用户复购商品数", "item_decay_slope": "商品热度趋势斜率",
    "user_avg_interval_hours": "用户平均行为间隔", "rfm_r_score": "RFM-R 最近购买",
    "rfm_f_score": "RFM-F 购买频次", "rfm_m_score": "RFM-M 购买品类广度",
    "user_category_pref_score": "用户类目偏好分", "item_category_te": "类目购买率编码",
}

BEHAVIOR_NOTE = "特征值为标准化（z-score）后的取值，0 代表全体均值水平"

# ========== 页面配置 ==========
st.set_page_config(page_title="模型评估看板", layout="wide")
st.title("🤖 看板二：模型评估·用户购买预测模型评估与解释看板")
st.caption(
    "💡 最终模型：**XGBoost Optuna**（PR-AUC 最优）| 分类阈值 **0.2**（内部调优验证集选定）| "
    "评估集：**测试集 test.parquet（468,316 对，正样本 449，正样本率 0.096%）** | "
    "所有特征均为标准化取值。模块 2/3 支持左侧全局筛选联动重算，模块 1/4/5 基于固定评估产出。"
)

# ================================================================
# 数据加载（全部缓存）
# ================================================================

@st.cache_data
def load_predictions():
    """加载测试集预测明细，并关联 test.parquet 的分层所需列（关联键 user_id+item_id，100% 匹配）。"""
    pred = pd.read_csv(os.path.join(EVAL_DIR, "step9_test_predictions.csv"))
    test = pd.read_parquet(
        os.path.join(OUT_DIR, "test.parquet"),
        columns=["user_id", "item_id", "last_time", "is_power_user",
                 "user_pv_count", "item_pv_count", "item_category_te"],
    )
    pred = pred.merge(test, on=["user_id", "item_id"], how="left")
    pred["date"] = pd.to_datetime(pred["last_time"]).dt.date
    return pred


@st.cache_data
def load_test_features():
    """加载测试集全部特征值（SHAP 蜂群图 / 依赖图需要未标准化的原始关联数据）。"""
    return pd.read_parquet(os.path.join(OUT_DIR, "test.parquet"))


@st.cache_data
def load_general_metrics():
    """最终模型测试集整体指标（步骤九产出）。"""
    return pd.read_csv(os.path.join(EVAL_DIR, "step9_generalization_metrics.csv")).iloc[0]


@st.cache_data
def load_threshold_metrics():
    """阈值敏感性扫描（步骤九产出，全量测试集，仅诊断用途）。"""
    return pd.read_csv(os.path.join(EVAL_DIR, "step9_threshold_metrics.csv"))


@st.cache_data
def load_model_comparison():
    """合并各阶段模型指标，构建多模型对比表。

    口径说明：除最后一行外，各模型指标均在【完整验证集 val（93.6万对）】上测得；
    xgboost_optuna_test 一行为最终模型在【测试集 test】上的泛化成绩（步骤九）。
    """
    rows = []

    # 1) 基线模型：传统 ML 三模型 + 序列模型三模型（验证集）
    base = pd.read_csv(os.path.join(OUT_DIR, "baseline_models", "baseline_metrics.csv"))
    for _, r in base.iterrows():
        rows.append({
            "模型": r["model"], "阶段": "步骤六 基线", "子阶段": "传统ML",
            "评估集": "验证集 val", "ROC-AUC": r["roc_auc"], "PR-AUC(AP)": r["pr_auc_ap"],
            "LogLoss": r["log_loss"], "F1": r["f1_at_0_5"],
            "选定阈值": r["selected_threshold"],
        })
    # GRU / LSTM 基线模型文件已不在当前 output，按 Part6 文档指标硬编码（固定验证集结果）
    rows.extend([
        {"模型": "gru", "阶段": "步骤六 基线", "子阶段": "序列模型", "评估集": "验证集 val",
         "ROC-AUC": 0.617809, "PR-AUC(AP)": 0.003618, "LogLoss": 0.060225,
         "F1": 0.013627, "选定阈值": 0.90},
        {"模型": "lstm", "阶段": "步骤六 基线", "子阶段": "序列模型", "评估集": "验证集 val",
         "ROC-AUC": 0.597754, "PR-AUC(AP)": 0.002506, "LogLoss": 0.182243,
         "F1": 0.006088, "选定阈值": 0.90},
    ])
    # DIN 基线（序列模型，验证集）
    with open(os.path.join(OUT_DIR, "part7_din_stacking", "din_final_metrics.json")) as f:
        din = json.load(f)["metrics"]
    rows.append({
        "模型": "din", "阶段": "步骤六 基线", "子阶段": "序列模型", "评估集": "验证集 val",
        "ROC-AUC": din["roc_auc"], "PR-AUC(AP)": din["pr_auc_ap"],
        "LogLoss": din["log_loss"], "F1": din["f1_at_0_5"], "选定阈值": din["best_threshold"],
    })

    # 2) Optuna 调优两模型（验证集）
    bo = pd.read_csv(os.path.join(OUT_DIR, "optuna_tuned_models", "baseline_vs_optuna_metrics.csv"))
    for _, r in bo[bo["source"] == "optuna_tuned"].iterrows():
        rows.append({
            "模型": r["model"], "阶段": "步骤七 Optuna 调优", "子阶段": "传统ML",
            "评估集": "验证集 val", "ROC-AUC": r["roc_auc"], "PR-AUC(AP)": r["pr_auc_ap"],
            "LogLoss": r["log_loss"], "F1": r["f1_at_0_5"],
            "选定阈值": r["selected_threshold"],
        })

    # 3) Tree 融合：等权融合 + OOF Stacking（验证集）
    stack = pd.read_csv(os.path.join(OUT_DIR, "tree_stacking", "stacking_validation_metrics.csv"))
    for model_name, display_name, sub_stage in [
        ("tree_equal", "tree_equal", "等权融合"),
        ("stacking", "oof_stacking", "OOF Stacking"),
    ]:
        r = stack[stack["model"] == model_name].iloc[0]
        rows.append({
            "模型": display_name, "阶段": "步骤七 Tree融合", "子阶段": sub_stage,
            "评估集": "验证集 val", "ROC-AUC": r["roc_auc"], "PR-AUC(AP)": r["pr_auc_ap"],
            "LogLoss": r["log_loss"], "F1": r["f1_at_threshold"],
            "选定阈值": r["threshold"],
        })

    # 4) DIN + Tree 三模型固定权重融合（验证集）
    din_tree = pd.read_csv(
        os.path.join(OUT_DIR, "part7_din_stacking", "stacking_comparison_full_validation.csv")
    )
    blend_map = {
        "tree_plus_din_equal": "DIN等权(1/3)",
        "tree_plus_din_20pct": "DIN权重20%",
        "tree_plus_din_40pct": "DIN权重40%",
    }
    for _, r in din_tree.iterrows():
        key = r["blend"]
        if key not in blend_map:
            continue
        rows.append({
            "模型": blend_map[key], "阶段": "步骤七 DIN+Tree", "子阶段": "固定权重",
            "评估集": "验证集 val", "ROC-AUC": r["roc_auc"], "PR-AUC(AP)": r["pr_auc_ap"],
            "LogLoss": r["log_loss"], "F1": r["f1_at_0_5"],
            "选定阈值": r["best_threshold"],
        })

    # 5) XGBoost + DIN 概率融合（验证集）——只取概率融合，rank 融合仅诊断用不展示
    xdb = pd.read_csv(os.path.join(OUT_DIR, "xgb_din_fusion", "xgb_din_fusion_metrics.csv"))
    xdb = xdb[xdb["score_type"] == "probability_blend"].copy()
    xdb["din_weight_pct"] = (xdb["din_weight"] * 100).astype(int)
    for _, r in xdb.iterrows():
        w = int(r["din_weight_pct"])
        rows.append({
            "模型": f"xgb_{100-w}%_din_{w}%", "阶段": "步骤七 XGB+DIN", "子阶段": f"DIN权重{w}%",
            "评估集": "验证集 val", "ROC-AUC": r["roc_auc"], "PR-AUC(AP)": r["pr_auc_ap"],
            "LogLoss": r["log_loss"], "F1": r["f1_at_fixed_threshold"],
            "选定阈值": r["fixed_threshold"],
        })

    # 6) 最终模型测试集成绩（步骤九）
    gm = load_general_metrics()
    rows.append({
        "模型": "xgboost_optuna ★", "阶段": "步骤九 最终泛化", "子阶段": "最终模型",
        "评估集": "测试集 test", "ROC-AUC": gm["roc_auc"], "PR-AUC(AP)": gm["pr_auc_ap"],
        "LogLoss": gm["log_loss"], "F1": gm["f1"], "选定阈值": gm["threshold"],
    })

    return pd.DataFrame(rows).sort_values("PR-AUC(AP)", ascending=False).reset_index(drop=True)


@st.cache_data
def compute_shap_sample(n_total=10000):
    """运行时重算 SHAP：加载 XGBoost Optuna 模型，对测试集采样（正样本全取+负样本补齐）
    用原生 pred_contribs 计算 SHAP 贡献值（log-odds 空间）。

    Returns:
        (shap_df, feat_df, ok):
        shap_df — 采样行 × 24 特征的 SHAP 值（+bias 列）
        feat_df — 对应行的特征值 / label / 概率（用于着色与局部解释）
        ok — 模型加载是否成功（失败则看板降级为读取已保存 CSV）
    """
    try:
        import joblib
        import xgboost as xgb

        test = load_test_features()
        # 采样：正样本全取，负样本随机补齐到 n_total
        pos = test[test["label"] == 1]
        neg = test[test["label"] == 0]
        n_neg = max(n_total - len(pos), 0)
        neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=42)
        sample = pd.concat([pos, neg_s], ignore_index=True)
        sample = sample.sample(frac=1.0, random_state=42).reset_index(drop=True)

        art = joblib.load(os.path.join(OUT_DIR, "optuna_tuned_models", "xgboost_optuna.joblib"))
        model = art["model"] if isinstance(art, dict) else art

        X = sample[FEATURE_COLS].to_numpy(dtype=np.float32)
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        shap_df = pd.DataFrame(contribs[:, :len(FEATURE_COLS)], columns=FEATURE_COLS)
        shap_df["__bias__"] = contribs[:, -1]

        # 预测概率（sigmoid(log-odds)）
        logit = contribs.sum(axis=1)
        sample["purchase_probability"] = 1.0 / (1.0 + np.exp(-logit))
        return shap_df, sample, True
    except Exception:
        return None, None, False


@st.cache_data
def load_saved_shap():
    """降级方案：读取步骤九已保存的 SHAP 样本明细（仅 SHAP 值 + label，无特征值）。"""
    return pd.read_csv(os.path.join(INTERP_DIR, "xgboost_native_shap_values_sample.csv"))


@st.cache_data
def load_shap_importance():
    return pd.read_csv(os.path.join(INTERP_DIR, "xgboost_native_shap_importance.csv"))


@st.cache_data
def load_error_profile():
    return pd.read_csv(os.path.join(INTERP_DIR, "error_feature_profile.csv"))


@st.cache_data
def load_segment_metrics():
    return pd.read_csv(os.path.join(INTERP_DIR, "error_segment_metrics.csv"))


@st.cache_data
def load_top_errors():
    return pd.read_csv(os.path.join(INTERP_DIR, "top_error_samples.csv"))


# ================================================================
# 侧边栏全局筛选（时间 / 用户分层 / 类目购买率分层）
# ================================================================
pred_all = load_predictions()

st.sidebar.header("⚙️ 全局筛选（作用于模块 2 / 3）")

# 时间范围
min_date, max_date = pred_all["date"].min(), pred_all["date"].max()
date_range = st.sidebar.date_input(
    "📅 时间范围（样本 last_time）",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# 用户分层（基于测试集特征：活跃度四分位 + 超级用户）
upv = pred_all["user_pv_count"]
q25, q50, q75 = upv.quantile([0.25, 0.50, 0.75])
segment_options = {
    "all": "全部用户",
    "power": "超级用户 (is_power_user)",
    "normal": "普通用户",
    "low": f"低活跃 (浏览数 < Q25={q25:.2f})",
    "mid": f"中活跃 (Q25~Q75)",
    "high": f"高活跃 (浏览数 > Q75={q75:.2f})",
}
seg = st.sidebar.selectbox(
    "👥 用户分层",
    options=list(segment_options.keys()),
    format_func=lambda x: segment_options[x],
)

# 类目购买率分层（数据无原始类目列，用目标编码 item_category_te 分箱替代）
te_q33, te_q66 = pred_all["item_category_te"].quantile([1 / 3, 2 / 3])
cat_options = {
    "all": "全部类目",
    "low_rate": f"低购买率类目 (TE < {te_q33:.2f})",
    "mid_rate": "中购买率类目",
    "high_rate": f"高购买率类目 (TE > {te_q66:.2f})",
}
cat_seg = st.sidebar.selectbox(
    "🏷️ 类目购买率分层（基于目标编码）",
    options=list(cat_options.keys()),
    format_func=lambda x: cat_options[x],
)

# 应用筛选
if len(date_range) == 2:
    d_start, d_end = date_range
else:
    d_start, d_end = min_date, max_date

mask = (pred_all["date"] >= d_start) & (pred_all["date"] <= d_end)
if seg == "power":
    mask &= pred_all["is_power_user"] == True
elif seg == "normal":
    mask &= pred_all["is_power_user"] == False
elif seg == "low":
    mask &= pred_all["user_pv_count"] < q25
elif seg == "mid":
    mask &= (pred_all["user_pv_count"] >= q25) & (pred_all["user_pv_count"] <= q75)
elif seg == "high":
    mask &= pred_all["user_pv_count"] > q75

if cat_seg == "low_rate":
    mask &= pred_all["item_category_te"] < te_q33
elif cat_seg == "mid_rate":
    mask &= (pred_all["item_category_te"] >= te_q33) & (pred_all["item_category_te"] <= te_q66)
elif cat_seg == "high_rate":
    mask &= pred_all["item_category_te"] > te_q66

df = pred_all[mask].copy()

st.sidebar.markdown("---")
st.sidebar.info(
    f"当前筛选：{len(df):,} 对 | 正样本 {int(df['y_true'].sum()):,}"
)


# ================================================================
# 工具函数
# ================================================================

def group_metrics(sub: pd.DataFrame, threshold=DEFAULT_THRESHOLD):
    """计算一组预测的常用指标（子群性能拆解用）。"""
    y = sub["y_true"].values
    p = sub["purchase_probability"].values
    yhat = (p >= threshold).astype(int)
    pos = int(y.sum())
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    res = {
        "样本数": len(sub), "正样本": pos,
        "召回率": tp / pos if pos > 0 else np.nan,
        "精确率": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "F1": (2 * tp / (2 * tp + fp + (pos - tp))) if pos > 0 else np.nan,
        "ROC-AUC": roc_auc_score(y, p) if 0 < pos < len(sub) else np.nan,
    }
    return res


st.markdown("---")

# ================================================================
# 模块一：模型核心指标概览区
# ================================================================
st.header("📈 模块一：模型核心指标概览")

gm = load_general_metrics()

st.subheader("📋 最终模型（XGBoost Optuna）测试集核心指标")
st.caption(
    f"分类阈值 **{gm['threshold']}**（来源：内部调优验证集选定，非测试集调参）| "
    f"测试集 {int(gm['n_rows']):,} 对，正样本 {int(gm['positive_count'])}（{gm['positive_rate']:.4%}）"
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎯 ROC-AUC", f"{gm['roc_auc']:.4f}", help="排序能力：模型把正样本排在前面的能力")
with c2:
    st.metric("⭐ PR-AUC (AP)", f"{gm['pr_auc_ap']:.4f}",
              help="极不平衡场景下的主指标，随机基线≈正样本率 0.096%")
with c3:
    st.metric("⚖️ LogLoss", f"{gm['log_loss']:.4f}")
with c4:
    st.metric("📌 当前阈值", f"{gm['threshold']}", help="阈值来源=internal_tuning_validation")
c5, c6, c7, c8 = st.columns(4)
with c5:
    st.metric("✅ 准确率", f"{gm['accuracy']:.4f}")
with c6:
    st.metric("🎯 精确率", f"{gm['precision']:.2%}", help="预测为购买中真实购买的比例")
with c7:
    st.metric("🔁 召回率", f"{gm['recall']:.2%}", help="真实购买中被捕捉到的比例")
with c8:
    st.metric("🏆 F1", f"{gm['f1']:.4f}")

st.markdown("---")

st.subheader("🆚 多模型横向对比")
st.caption(
    "口径说明：除最后一行外，各模型指标均在【验证集 val（93.6万对）】上测得；"
    "★ 为最终模型在【测试集 test】上的泛化成绩。PR-AUC 为主选型指标（不平衡场景）。"
)

comp = load_model_comparison()

# 颜色：按阶段分组，同一阶段同色系；最终模型红色高亮
stage_colors = {
    "步骤六 基线": "#1f77b4",
    "步骤七 Optuna 调优": "#2ca02c",
    "步骤七 Tree融合": "#9467bd",
    "步骤七 DIN+Tree": "#ff7f0e",
    "步骤七 XGB+DIN": "#8c564b",
    "步骤九 最终泛化": "#d62728",
}
comp["颜色"] = comp["阶段"].map(stage_colors)
comp["颜色"] = comp["颜色"].fillna("#7f7f7f")

# 中文显示名
cn_map = {
    "logistic_regression": "逻辑回归", "lightgbm": "LightGBM基线",
    "xgboost": "XGBoost基线", "gru": "GRU",
    "lstm": "LSTM", "din": "DIN",
    "lightgbm_optuna": "LightGBM调优", "xgboost_optuna": "XGBoost调优",
    "tree_equal": "Tree等权", "oof_stacking": "OOF Stacking",
    "DIN等权(1/3)": "DIN等权", "DIN权重20%": "DIN20%",
    "DIN权重40%": "DIN40%", "xgb_100%_din_0%": "XGBoost",
    "xgb_95%_din_5%": "XGB95%+DIN5%", "xgb_90%_din_10%": "XGB90%+DIN10%",
    "xgb_80%_din_20%": "XGB80%+DIN20%", "xgb_70%_din_30%": "XGB70%+DIN30%",
    "xgb_50%_din_50%": "XGB50%+DIN50%", "xgboost_optuna ★": "XGBoost Optuna★",
}
comp["显示名"] = comp["模型"].map(cn_map).fillna(comp["模型"])

# 柱状图：ROC-AUC 与 PR-AUC 双面板
fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("ROC-AUC（排序能力）", "PR-AUC / AP（不平衡主指标）"))
for col, metric, fmt in [(1, "ROC-AUC", ".3f"), (2, "PR-AUC(AP)", ".3f")]:
    fig_comp.add_trace(
        go.Bar(
            x=comp["显示名"], y=comp[metric], marker_color=comp["颜色"],
            text=[f"{v:{fmt}}" for v in comp[metric]],
            textposition="outside", showlegend=False,
            customdata=np.stack([comp["评估集"], comp["阶段"], comp["子阶段"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b><br>" + metric + ": %{y:.4f}<br>"
                "评估集: %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra></extra>"
            ),
        ), row=1, col=col,
    )
fig_comp.update_layout(
    height=460, margin=dict(t=70, b=100),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
)
fig_comp.update_xaxes(tickangle=-35, tickfont=dict(size=10))
fig_comp.update_yaxes(title_text="ROC-AUC", range=[0, 1.08], row=1, col=1)
fig_comp.update_yaxes(title_text="PR-AUC", range=[0, max(0.18, comp["PR-AUC(AP)"].max() * 1.25)], row=1, col=2)
st.plotly_chart(fig_comp, use_container_width=True)

# 阶段图例（手动构建，避免重复）
legend_items = [f"<span style='color:{c}'>■</span> {s}" for s, c in stage_colors.items()]
st.markdown(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(legend_items), unsafe_allow_html=True)

disp = comp[["显示名", "阶段", "子阶段", "评估集", "ROC-AUC", "PR-AUC(AP)", "LogLoss", "F1", "选定阈值"]].copy()
disp = disp.rename(columns={"显示名": "模型"})
for c_ in ["ROC-AUC", "PR-AUC(AP)", "LogLoss", "F1"]:
    disp[c_] = disp[c_].map(lambda v: f"{v:.4f}")
disp["选定阈值"] = disp["选定阈值"].map(lambda v: f"{v:g}")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.success(
    "🎯 **选型结论**：XGBoost Optuna 在验证集以 PR-AUC **0.1560** 领先全部 15+ 个候选方案。"
    "关键对比：LightGBM 调优后 PR-AUC 反降（0.1249→0.1050），Tree 等权/OOF Stacking 接近但未超越，"
    "所有 DIN 融合方案（DIN 等权、DIN20%、DIN40%、XGB+DIN 系列）均低于 XGBoost Optuna 单模型。"
    "因此最终定版为 XGBoost Optuna 单模型，并在测试集取得 PR-AUC **0.1312** / ROC-AUC **0.9942** 的泛化成绩。"
)

st.markdown("---")

# ================================================================
# 模块二：整体性能深度评估区
# ================================================================
st.header("🔬 模块二：整体性能深度评估")
st.caption("💡 本模块受左侧全局筛选联动重算（当前为筛选后子集）；阈值敏感性曲线为全量测试集固定诊断结果。")

if df["y_true"].nunique() < 2 or len(df) == 0:
    st.warning("当前筛选条件下正/负样本不足，无法计算曲线。请放宽筛选。")
else:
    y_true = df["y_true"].values
    y_prob = df["purchase_probability"].values

    # ---- 2.1 ROC + PR 曲线 ----
    st.subheader("1️⃣ ROC 曲线 + PR 曲线组合")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ap_val = average_precision_score(y_true, y_prob)
    pos_rate = y_true.mean()

    fig_curves = make_subplots(rows=1, cols=2, subplot_titles=(
        f"ROC 曲线（AUC = {auc_val:.4f}）", f"PR 曲线（AP = {ap_val:.4f}）"))

    fig_curves.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name="最终模型",
        line=dict(color="#d62728", width=2.5),
        hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig_curves.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="随机基线",
        line=dict(color="gray", dash="dash", width=1.5),
    ), row=1, col=1)

    fig_curves.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name="最终模型",
        line=dict(color="#d62728", width=2.5), showlegend=False,
        hovertemplate="召回率: %{x:.4f}<br>精确率: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig_curves.add_trace(go.Scatter(
        x=[0, 1], y=[pos_rate, pos_rate], mode="lines", name=f"随机基线 (正样本率 {pos_rate:.3%})",
        line=dict(color="gray", dash="dash", width=1.5),
    ), row=1, col=2)

    fig_curves.update_xaxes(title_text="假阳性率 FPR", range=[0, 1], row=1, col=1)
    fig_curves.update_yaxes(title_text="真阳性率 TPR", range=[0, 1.02], row=1, col=1)
    fig_curves.update_xaxes(title_text="召回率 Recall", range=[0, 1], row=1, col=2)
    fig_curves.update_yaxes(title_text="精确率 Precision", range=[0, 1.02], row=1, col=2)
    fig_curves.update_layout(height=430, legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.3))
    st.plotly_chart(fig_curves, use_container_width=True)

    st.caption(
        f"📌 解读：ROC-AUC {auc_val:.4f} 表明排序能力极强；PR-AUC {ap_val:.4f} "
        f"远高于随机水平（{pos_rate:.4%}），但绝对值受 0.1% 正样本率制约——"
        "这是极不平衡场景的正常形态，业务上应配合 Top-K 策略使用（见模块三）。"
    )

    # ---- 2.2 阈值敏感性分析 ----
    st.subheader("2️⃣ 阈值敏感性分析")
    th = load_threshold_metrics()
    best_f1_idx = th["f1"].idxmax()
    best_row = th.loc[best_f1_idx]

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(
        x=th["threshold"], y=th["precision"], mode="lines+markers", name="精确率",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="阈值 %{x:.2f}<br>精确率 %{y:.2%}<extra></extra>"))
    fig_th.add_trace(go.Scatter(
        x=th["threshold"], y=th["recall"], mode="lines+markers", name="召回率",
        line=dict(color="#d62728", width=2.5),
        hovertemplate="阈值 %{x:.2f}<br>召回率 %{y:.2%}<extra></extra>"))
    fig_th.add_trace(go.Scatter(
        x=th["threshold"], y=th["f1"], mode="lines+markers", name="F1",
        line=dict(color="#2ca02c", width=2.5),
        hovertemplate="阈值 %{x:.2f}<br>F1 %{y:.4f}<extra></extra>"))

    fig_th.add_vline(x=DEFAULT_THRESHOLD, line_dash="dot", line_color="gray",
                     annotation_text=f"当前阈值 {DEFAULT_THRESHOLD}", annotation_position="top left")
    fig_th.add_vline(x=best_row["threshold"], line_dash="dash", line_color="#2ca02c",
                     annotation_text=f"F1最优 {best_row['threshold']:.2f}", annotation_position="top right")
    fig_th.add_annotation(
        x=best_row["threshold"], y=best_row["f1"],
        text=f"F1={best_row['f1']:.3f}", showarrow=True, arrowhead=2, font=dict(color="#2ca02c"))

    fig_th.update_layout(
        title="分类阈值 × 精确率 / 召回率 / F1（全量测试集诊断）",
        xaxis=dict(title="分类阈值"), yaxis=dict(title="指标值", tickformat=".2%"),
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig_th, use_container_width=True)

    st.caption(
        f"📌 阈值 {DEFAULT_THRESHOLD}（内部验证集选定）在测试集上 F1={th[th['threshold'] == DEFAULT_THRESHOLD]['f1'].iloc[0]:.4f}；"
        f"测试集诊断最优点为 {best_row['threshold']:.2f}（F1={best_row['f1']:.4f}）。"
        "按防泄漏协议，测试集诊断结果不反向用于调参，仅验证选定阈值的合理性。"
    )

    # ---- 2.3 归一化混淆矩阵 ----
    st.subheader("3️⃣ 归一化混淆矩阵")
    y_hat = (y_prob >= DEFAULT_THRESHOLD).astype(int)
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    tp = int(((y_true == 1) & (y_hat == 1)).sum())

    labels = ["未购买 (0)", "购买 (1)"]
    cm_abs = np.array([[tn, fp], [fn, tp]])
    cm_norm = cm_abs / cm_abs.sum(axis=1, keepdims=True) * 100

    col_l, col_r = st.columns(2)
    with col_l:
        fig_cm_abs = go.Figure(go.Heatmap(
            z=cm_abs, x=labels, y=labels,
            colorscale="Blues", showscale=False,
            text=[[f"TN {tn:,}", f"FP {fp:,}"], [f"FN {fn:,}", f"TP {tp:,}"]],
            texttemplate="%{text}", textfont=dict(size=15),
            hovertemplate="真实 %{y} | 预测 %{x}<br>数量 %{z:,}<extra></extra>"))
        fig_cm_abs.update_layout(title="绝对数值", xaxis_title="预测", yaxis_title="真实",
                                 height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm_abs, use_container_width=True)
    with col_r:
        fig_cm_n = go.Figure(go.Heatmap(
            z=cm_norm, x=labels, y=labels,
            colorscale="RdYlGn", zmin=0, zmax=100,
            colorbar=dict(title="占比 %", ticksuffix="%"),
            text=[[f"TN {cm_norm[0, 0]:.1f}%", f"FP {cm_norm[0, 1]:.1f}%"],
                  [f"FN {cm_norm[1, 0]:.1f}%", f"TP {cm_norm[1, 1]:.1f}%"]],
            texttemplate="%{text}", textfont=dict(size=15),
            hovertemplate="真实 %{y} | 预测 %{x}<br>行内占比 %{z:.2f}%<extra></extra>"))
        fig_cm_n.update_layout(title="行归一化（每行合计 100%）", xaxis_title="预测", yaxis_title="真实",
                               height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm_n, use_container_width=True)

    st.caption(
        f"📌 真实购买用户被召回 **{cm_norm[1, 1]:.2%}**（TP/{cm_abs[1].sum()}）；"
        f"真实未购买用户被误判 **{cm_norm[0, 1]:.2%}**（FP/{cm_abs[0].sum()}）。"
        "行归一化视角可摆脱类别不平衡对绝对数量的掩盖。"
    )

    # ---- 2.4 概率校准曲线 ----
    st.subheader("4️⃣ 概率校准曲线（可靠性曲线）")
    bins = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    cal = pd.DataFrame({"p": y_prob, "y": y_true})
    cal["bin"] = pd.cut(cal["p"], bins=bins, right=False, include_lowest=True)
    cal_g = cal.groupby("bin", observed=True).agg(
        n=("y", "size"), pred_mean=("p", "mean"), true_rate=("y", "mean")).reset_index()
    cal_g = cal_g[cal_g["n"] >= 30]  # 过滤小样本箱
    cal_g["bin_label"] = cal_g["bin"].astype(str)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="完美校准",
        line=dict(color="gray", dash="dash", width=1.5)))
    fig_cal.add_trace(go.Scatter(
        x=cal_g["pred_mean"], y=cal_g["true_rate"], mode="markers+lines", name="模型",
        line=dict(color="#d62728", width=2.5),
        marker=dict(size=14, color="#d62728",
                    line=dict(width=1.5, color="white")),
        customdata=np.stack([cal_g["bin_label"], cal_g["n"]], axis=1),
        hovertemplate="概率箱 %{customdata[0]}<br>平均预测概率 %{x:.4f}<br>"
                      "真实购买率 %{y:.3%}<br>样本量 %{customdata[1]:,}<extra></extra>"))
    fig_cal.update_layout(
        title="预测概率 vs 真实购买率（每箱 ≥30 样本）",
        xaxis=dict(title="模型平均预测概率", tickformat=".2%", type="log"),
        yaxis=dict(title="箱内真实购买率", tickformat=".3%"),
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig_cal, use_container_width=True)

    st.caption(
        "📌 横轴为对数坐标。模型概率整体贴合对角线则输出可信；"
        "低概率区间的偏离主要来自极不平衡下的稀疏正样本，使用时应更关注概率的相对排序而非绝对值。"
    )

st.markdown("---")

# ================================================================
# 模块三：多维度分群性能拆解区
# ================================================================
st.header("🧩 模块三：多维度分群性能拆解")
st.caption("💡 定位模型的优势场景与短板场景：Top-K 推荐效果、用户/商品分层、时间稳定性。受全局筛选联动。")

if df["y_true"].nunique() < 2 or len(df) == 0:
    st.warning("当前筛选条件下数据不足，模块三不可用。")
else:
    # ---- 3.1 Top-K 推荐效果 ----
    st.subheader("1️⃣ Top-K 推荐效果折线图")
    st.caption(
        "💡 按用户内概率排名（rank_in_user）取每位用户 Top-K 候选商品。"
        "Precision@K = Top-K 命中数 / Top-K 总数；Recall@K = Top-K 命中数 / 全部正样本。"
    )
    ks = [5, 10, 20, 30, 50]
    prec_k, rec_k, map_k = [], [], []
    user_n_with_k = []
    for k in ks:
        topk = df[df["rank_in_user"] <= k].copy()
        hits = int(topk["y_true"].sum())
        prec_k.append(hits / len(topk) if len(topk) else np.nan)
        rec_k.append(hits / df["y_true"].sum() if df["y_true"].sum() else np.nan)
        user_n_with_k.append(topk["user_id"].nunique())
        # MAP@K（向量化）：组内按 rank 排序，AP@K = Σ(precision@i × hit_i) / 组内正样本数
        topk = topk.sort_values(["user_id", "rank_in_user"])
        topk["_rank"] = topk.groupby("user_id").cumcount() + 1
        topk["_cum_hits"] = topk.groupby("user_id")["y_true"].cumsum()
        topk["_prec_at_i"] = topk["_cum_hits"] / topk["_rank"]
        topk["_ap_num"] = topk["_prec_at_i"] * topk["y_true"]
        ap_sum = topk.groupby("user_id").agg(
            num=("_ap_num", "sum"), pos=("_cum_hits", "max"))
        ap_sum["ap"] = np.where(ap_sum["pos"] > 0, ap_sum["num"] / ap_sum["pos"], 0.0)
        map_k.append(ap_sum["ap"].mean() if len(ap_sum) else np.nan)

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=ks, y=prec_k, mode="lines+markers", name="Precision@K",
        line=dict(color="#d62728", width=2.5),
        text=[f"{v:.3%}" for v in prec_k], textposition="top center",
        hovertemplate="K=%{x}<br>Precision@K=%{y:.3%}<extra></extra>"))
    fig_k.add_trace(go.Scatter(
        x=ks, y=rec_k, mode="lines+markers", name="Recall@K",
        line=dict(color="#1f77b4", width=2.5),
        text=[f"{v:.2%}" for v in rec_k], textposition="bottom center",
        hovertemplate="K=%{x}<br>Recall@K=%{y:.2%}<extra></extra>"))
    fig_k.add_trace(go.Scatter(
        x=ks, y=map_k, mode="lines+markers", name="MAP@K",
        line=dict(color="#2ca02c", width=2.5),
        text=[f"{v:.3f}" for v in map_k], textposition="top center",
        hovertemplate="K=%{x}<br>MAP@K=%{y:.4f}<extra></extra>"))
    fig_k.update_layout(
        title="Top-K 推荐效果（对数横轴）",
        xaxis=dict(title="K 值", type="log", tickvals=ks, ticktext=[str(k) for k in ks]),
        yaxis=dict(title="指标值", tickformat=".2%"),
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig_k, use_container_width=True)

    st.caption(
        f"📌 例：Top-10 的命中精确率约 {prec_k[1]:.3%}，是全体正样本率"
        f"（{df['y_true'].mean():.4%}）的约 {prec_k[1] / df['y_true'].mean():.0f} 倍——"
        "模型的排序价值体现在把购买对集中到头部候选。"
    )

    # ---- 3.2 用户分层性能对比 ----
    st.subheader("2️⃣ 用户分层性能对比")
    seg_defs = [
        ("超级用户", df[df["is_power_user"] == True]),
        ("普通用户", df[df["is_power_user"] == False]),
        ("低活跃(浏览Q1)", df[df["user_pv_count"] <= q25]),
        ("中活跃(Q2-Q3)", df[(df["user_pv_count"] > q25) & (df["user_pv_count"] <= q75)]),
        ("高活跃(Q4)", df[df["user_pv_count"] > q75]),
    ]
    seg_rows = []
    for name, sub in seg_defs:
        if len(sub) == 0:
            continue
        m = group_metrics(sub)
        m["群体"] = name
        seg_rows.append(m)
    seg_df = pd.DataFrame(seg_rows)[["群体", "样本数", "正样本", "ROC-AUC", "召回率", "精确率", "F1"]]

    fig_user = make_subplots(specs=[[{"secondary_y": True}]])
    fig_user.add_trace(go.Bar(
        x=seg_df["群体"], y=seg_df["ROC-AUC"], name="ROC-AUC",
        marker_color="#1f77b4", opacity=0.85,
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in seg_df["ROC-AUC"]],
        textposition="outside",
        customdata=np.stack([seg_df["样本数"], seg_df["正样本"]], axis=1),
        hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.4f}<br>样本数: %{customdata[0]:,}<br>正样本: %{customdata[1]}<extra></extra>"), secondary_y=False)
    fig_user.add_trace(go.Scatter(
        x=seg_df["群体"], y=seg_df["召回率"], name="召回率", mode="lines+markers",
        line=dict(color="#d62728", width=3), marker=dict(size=11),
        text=[f"{v:.1%}" if pd.notna(v) else "N/A" for v in seg_df["召回率"]],
        textposition="top center",
        hovertemplate="召回率: %{y:.2%}<extra></extra>"), secondary_y=True)
    fig_user.update_layout(
        title="用户分层 × ROC-AUC / 召回率（阈值 0.2）",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    fig_user.update_yaxes(title_text="ROC-AUC", range=[0, 1.1], secondary_y=False)
    fig_user.update_yaxes(title_text="召回率", tickformat=".0%", range=[0, 1.1], secondary_y=True)
    st.plotly_chart(fig_user, use_container_width=True)

    st.markdown("**📊 用户分层数值明细**")
    disp_seg = seg_df.copy()
    for c_ in ["ROC-AUC", "召回率", "精确率", "F1"]:
        disp_seg[c_] = disp_seg[c_].map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")
    st.dataframe(disp_seg, use_container_width=True, hide_index=True)

    # ---- 3.3 商品分层性能对比 ----
    st.subheader("3️⃣ 商品分层性能对比")
    iq1, iq2 = df["item_pv_count"].quantile([1 / 3, 2 / 3])
    item_defs = [
        ("爆款商品(浏览Top1/3)", df[df["item_pv_count"] > iq2]),
        ("腰部商品(中1/3)", df[(df["item_pv_count"] > iq1) & (df["item_pv_count"] <= iq2)]),
        ("长尾商品(浏览Bottom1/3)", df[df["item_pv_count"] <= iq1]),
    ]
    item_rows = []
    for name, sub in item_defs:
        if len(sub) == 0:
            continue
        m = group_metrics(sub)
        m["群体"] = name
        item_rows.append(m)
    item_df = pd.DataFrame(item_rows)[["群体", "样本数", "正样本", "ROC-AUC", "召回率", "精确率", "F1"]]

    fig_item_seg = make_subplots(specs=[[{"secondary_y": True}]])
    fig_item_seg.add_trace(go.Bar(
        x=item_df["群体"], y=item_df["ROC-AUC"], name="ROC-AUC",
        marker_color="#1f77b4", opacity=0.85,
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in item_df["ROC-AUC"]],
        textposition="outside",
        customdata=np.stack([item_df["样本数"], item_df["正样本"]], axis=1),
        hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.4f}<br>样本数: %{customdata[0]:,}<br>正样本: %{customdata[1]}<extra></extra>"), secondary_y=False)
    fig_item_seg.add_trace(go.Scatter(
        x=item_df["群体"], y=item_df["召回率"], name="召回率", mode="lines+markers",
        line=dict(color="#d62728", width=3), marker=dict(size=11),
        text=[f"{v:.1%}" if pd.notna(v) else "N/A" for v in item_df["召回率"]],
        textposition="top center"), secondary_y=True)
    fig_item_seg.update_layout(
        title="商品分层 × ROC-AUC / 召回率（阈值 0.2）",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    fig_item_seg.update_yaxes(title_text="ROC-AUC", range=[0, 1.1], secondary_y=False)
    fig_item_seg.update_yaxes(title_text="召回率", tickformat=".0%", range=[0, 1.1], secondary_y=True)
    st.plotly_chart(fig_item_seg, use_container_width=True)

    st.markdown("**📊 商品分层数值明细**")
    disp_item = item_df.copy()
    for c_ in ["ROC-AUC", "召回率", "精确率", "F1"]:
        disp_item[c_] = disp_item[c_].map(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A")
    st.dataframe(disp_item, use_container_width=True, hide_index=True)

    # ---- 3.4 时间维度稳定性 ----
    st.subheader("4️⃣ 时间维度稳定性趋势图")
    daily_rows = []
    for d, sub in df.groupby("date"):
        if sub["y_true"].nunique() < 2:
            continue
        m = group_metrics(sub)
        m["日期"] = d
        daily_rows.append(m)
    daily = pd.DataFrame(daily_rows)

    if not daily.empty:
        fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
        fig_daily.add_trace(go.Scatter(
            x=daily["日期"], y=daily["ROC-AUC"], mode="lines+markers", name="每日 ROC-AUC",
            line=dict(color="#1f77b4", width=2.5), marker=dict(size=7),
            customdata=np.stack([daily["样本数"], daily["正样本"]], axis=1),
            hovertemplate="%{x}<br>AUC: %{y:.4f}<br>样本数: %{customdata[0]:,}<br>正样本: %{customdata[1]}<extra></extra>"), secondary_y=False)
        fig_daily.add_trace(go.Scatter(
            x=daily["日期"], y=daily["召回率"], mode="lines+markers", name="每日召回率",
            line=dict(color="#d62728", width=2.5), marker=dict(size=7),
            hovertemplate="%{x}<br>召回率: %{y:.2%}<extra></extra>"), secondary_y=True)

        # 12-12 大促标注
        promo = pd.Timestamp("2025-12-12").date()
        if daily["日期"].min() <= promo <= daily["日期"].max():
            fig_daily.add_vrect(
                x0=promo, x1=promo + pd.Timedelta(days=1),
                fillcolor="#ff7f0e", opacity=0.12, line_width=0,
                annotation_text="双12大促", annotation_position="top left")

        fig_daily.update_layout(
            title="测试集每日 ROC-AUC 与召回率（阈值 0.2）",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
        fig_daily.update_yaxes(title_text="ROC-AUC", secondary_y=False, range=[0.5, 1.02])
        fig_daily.update_yaxes(title_text="召回率", tickformat=".0%", secondary_y=True, range=[0, 1.05])
        st.plotly_chart(fig_daily, use_container_width=True)

        mean_auc = daily["ROC-AUC"].mean()
        std_auc = daily["ROC-AUC"].std()
        st.caption(
            f"📌 每日 ROC-AUC 均值 **{mean_auc:.4f}**，标准差 **{std_auc:.4f}**。"
            f"测试集每日正样本仅 {daily['正样本'].min():.0f}~{daily['正样本'].max():.0f} 个，"
            "召回率日间波动主要来自稀疏正样本的抽样噪声，AUC 稳定在 0.98 以上即视为时间稳定。"
        )
    else:
        st.info("筛选范围内无可计算的时间分段。")

st.markdown("---")

# ================================================================
# 模块四：特征可解释性分析区
# ================================================================
st.header("🔍 模块四：特征可解释性分析")
st.caption(
    "💡 基于步骤九 XGBoost 原生 SHAP（pred_contribs）。看板启动时对测试集采样"
    "（正样本全取 + 负样本随机补齐 ≈ 1 万条）实时重算。SHAP 值为 log-odds 空间的贡献。"
    + BEHAVIOR_NOTE + "。本模块不受全局筛选影响（固定诊断）。"
)

shap_df, feat_df, shap_ok = compute_shap_sample()
imp_df = load_shap_importance()

if not shap_ok:
    st.warning(
        "⚠️ 无法加载 XGBoost 模型（当前 Python 环境缺 xgboost 运行库），"
        "已降级为读取步骤九保存的 SHAP 样本明细（无特征值着色）。"
        "如需完整蜂群图/依赖图，请在装有 xgboost 的训练环境中运行本看板。"
    )
    saved = load_saved_shap()
    # ---- 降级：SHAP 值分布（按 label 着色） ----
    st.subheader("1️⃣ SHAP 特征重要性（Top 10）")
    top10 = imp_df.nlargest(10, "mean_abs_shap")
    fig_imp = go.Figure(go.Bar(
        x=top10["mean_abs_shap"], y=[FEATURE_CN.get(f, f) for f in top10["feature"]],
        orientation="h", marker_color="#1f77b4",
        text=[f"{v:.2f}" for v in top10["mean_abs_shap"]], textposition="outside"))
    fig_imp.update_layout(height=450, margin=dict(l=150),
                          xaxis_title="平均 |SHAP| 贡献（log-odds）")
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    # ---- 4.1 SHAP 蜂群图（Top 10） ----
    st.subheader("1️⃣ SHAP 特征重要性蜂群图（Top 10）")
    st.caption(
        "💡 纵轴按重要性排序，横轴为 SHAP 值（右=推向购买，左=推向不购买），"
        "颜色代表特征值高低（红=高、蓝=低，标准化值）。三重信息：重要性、方向、数值规律。"
    )
    mean_abs = shap_df[FEATURE_COLS].abs().mean().sort_values(ascending=False)
    top10_feats = mean_abs.head(10).index.tolist()

    rng = np.random.default_rng(42)
    fig_swarm = go.Figure()
    for i, feat in enumerate(top10_feats):
        sv = shap_df[feat].values
        fv = feat_df[feat].values
        # 下采样至每特征 4000 点，避免渲染卡顿
        if len(sv) > 4000:
            idx = rng.choice(len(sv), 4000, replace=False)
            sv, fv = sv[idx], fv[idx]
        jitter = rng.uniform(-0.22, 0.22, len(sv))
        fig_swarm.add_trace(go.Scattergl(
            x=sv, y=np.full(len(sv), i) + jitter,
            mode="markers",
            marker=dict(size=4, color=fv, opacity=0.55),
            name=feat,
            customdata=fv,
            hovertemplate=(f"<b>{FEATURE_CN.get(feat, feat)}</b><br>"
                           "SHAP: %{x:.3f}<br>特征值(标准化): %{customdata:.2f}<extra></extra>"),
            showlegend=False,
        ))

    fig_swarm.update_layout(
        height=520,
        xaxis=dict(title="SHAP 值（log-odds 贡献）→ 推向购买", zeroline=True, zerolinewidth=1, zerolinecolor="gray"),
        yaxis=dict(
            title="",
            tickvals=list(range(10)),
            ticktext=[f"{FEATURE_CN.get(f, f)}" for f in top10_feats],
            autorange="reversed"),
        coloraxis=dict(colorscale="RdBu_r", cmin=-2.5, cmax=2.5,
                       colorbar=dict(title="特征值<br>(标准化)")),
    )
    # 统一颜色轴：特征值高=红（暖），低=蓝（冷）；marker.color 存特征值，coloraxis 统一映射
    for tr in fig_swarm.data:
        tr.marker.coloraxis = "coloraxis"
    st.plotly_chart(fig_swarm, use_container_width=True)

    # ---- 4.2 SHAP 依赖图（Top 3） ----
    st.subheader("2️⃣ 核心特征 SHAP 依赖图（Top 3）")
    st.caption("💡 横轴为特征值（标准化），纵轴为 SHAP 值，挖掘非线性影响规律（拐点/饱和效应）。")
    top3_feats = mean_abs.head(3).index.tolist()

    fig_dep = make_subplots(rows=1, cols=3,
                            subplot_titles=[FEATURE_CN.get(f, f) for f in top3_feats])
    for col_i, feat in enumerate(top3_feats, start=1):
        fv = feat_df[feat].values
        sv = shap_df[feat].values
        if len(sv) > 4000:
            idx = rng.choice(len(sv), 4000, replace=False)
            fv, sv = fv[idx], sv[idx]
        fig_dep.add_trace(go.Scattergl(
            x=fv, y=sv, mode="markers",
            marker=dict(size=5, color=sv, colorscale="RdBu_r", cmin=-3, cmax=3, opacity=0.5),
            showlegend=False,
            hovertemplate=f"<b>{FEATURE_CN.get(feat, feat)}</b><br>"
                          "特征值: %{x:.2f}<br>SHAP: %{y:.3f}<extra></extra>",
        ), row=1, col=col_i)
        fig_dep.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=1, col=col_i)
    fig_dep.update_layout(height=420, showlegend=False)
    fig_dep.update_xaxes(title_text="特征值（标准化）")
    fig_dep.update_yaxes(title_text="SHAP 值")
    st.plotly_chart(fig_dep, use_container_width=True)

    # ---- 4.4 典型用户局部解释 ----
    st.subheader("3️⃣ 典型样本局部解释（瀑布图）")
    st.caption(
        "💡 选取三类典型样本：模型最自信的命中（TP）、最自信的误判（FP）、"
        "最严重的漏判（FN）。展示单个预测的特征贡献拆解（log-odds 空间）。"
    )
    sample_view = feat_df.copy()
    sample_view["__prob__"] = sample_view["purchase_probability"]

    tp_pick = sample_view[(sample_view["label"] == 1)].nlargest(1, "__prob__").iloc[0]
    fp_pick = sample_view[(sample_view["label"] == 0)].nlargest(1, "__prob__").iloc[0]
    fn_pick = sample_view[(sample_view["label"] == 1)].nsmallest(1, "__prob__").iloc[0]

    picks = [("✅ 自信命中 TP（正样本·高概率）", tp_pick),
             ("⚠️ 自信误判 FP（负样本·高概率）", fp_pick),
             ("❌ 严重漏判 FN（正样本·低概率）", fn_pick)]

    tab1, tab2, tab3 = st.tabs([p[0] for p in picks])
    for tab, (title, row) in zip([tab1, tab2, tab3], picks):
        with tab:
            idx = row.name
            s_row = shap_df.loc[idx]
            contribs = s_row[FEATURE_COLS].astype(float)
            top_c = contribs.abs().sort_values(ascending=False).head(8)
            bias = float(s_row["__bias__"])

            wf_labels = ["基准值 bias"] + \
                [f"{FEATURE_CN.get(f, f)}<br>({row[f]:+.2f})" for f in top_c.index] + \
                ["其他特征合计", "预测 logit"]
            wf_vals = [bias] + list(top_c.values) + \
                [float(contribs.sum() - top_c.sum()), 0]

            fig_wf = go.Figure(go.Waterfall(
                x=wf_labels, y=wf_vals,
                measure=["absolute"] + ["relative"] * (len(wf_vals) - 2) + ["total"],
                text=[f"{v:+.2f}" for v in wf_vals], textposition="outside",
                connector=dict(line=dict(color="lightgray")),
                increasing=dict(marker=dict(color="#d62728")),
                decreasing=dict(marker=dict(color="#1f77b4")),
                totals=dict(marker=dict(color="#2ca02c")),
            ))
            fig_wf.update_layout(
                title=(f"{title} | user {int(row['user_id']):,} · item {int(row['item_id']):,} | "
                       f"预测概率 {row['__prob__']:.3%} | 真实标签 {int(row['label'])}"),
                yaxis_title="log-odds 贡献", height=430,
                margin=dict(b=110))
            st.plotly_chart(fig_wf, use_container_width=True)

# ---- 正负向影响特征汇总（两种模式均可用） ----
st.subheader("4️⃣ 正负向影响特征汇总")
st.caption(
    "💡 基于正/负标签分组的 SHAP 均值对比：正值（暖色）= 该特征在真实购买样本上更推高预测，"
    "负值（冷色）= 在未购买样本上更推低预测。面向业务视角，无需算法基础。"
)
imp_sorted = imp_df.sort_values("positive_minus_negative_mean_shap", ascending=False)
contrast_col = "positive_minus_negative_mean_shap"
top_pos = imp_sorted.head(10)
top_neg = imp_sorted.tail(5).iloc[::-1]
plot_df = pd.concat([top_pos, top_neg])

fig_dir = go.Figure(go.Bar(
    x=plot_df[contrast_col],
    y=[FEATURE_CN.get(f, f) for f in plot_df["feature"]],
    orientation="h",
    marker=dict(color=np.where(plot_df[contrast_col] >= 0, "#d62728", "#1f77b4")),
    text=[f"{v:+.2f}" for v in plot_df[contrast_col]], textposition="outside",
    customdata=np.stack([plot_df["mean_abs_shap"], plot_df["contrast_direction"]], axis=1),
    hovertemplate="<b>%{y}</b><br>正负对比差: %{x:+.2f}<br>平均|SHAP|: %{customdata[0]:.2f}<br>"
                  "方向: %{customdata[1]}<extra></extra>"))
fig_dir.update_layout(
    title="Top10 正向促进 + Top5 负向抑制特征（正/负标签 SHAP 均值差）",
    xaxis_title="正样本均值SHAP − 负样本均值SHAP（log-odds）",
    height=max(480, 32 * len(plot_df) + 100), margin=dict(l=160))
fig_dir.add_vline(x=0, line_width=1, line_color="gray")
st.plotly_chart(fig_dir, use_container_width=True)

st.caption(
    "📌 例：商品购买数（item_buy_count）正负差最大——真实购买样本中该特征显著推高预测，"
    "与业务直觉一致：被买得多的商品更可能再次被买。"
)

st.markdown("---")

# ================================================================
# 模块五：错误分析与模型边界区
# ================================================================
st.header("🚨 模块五：错误分析与模型边界")
st.caption("💡 体现分析严谨性：错误样本画像、冷启动/长尾专项对比、高置信误判明细。基于步骤九固定产出。")

# ---- 5.1 错误样本画像对比 ----
st.subheader("1️⃣ 错误样本画像对比图")
prof = load_error_profile()

# 与全体均值的对比（全体=0 基线，画像为标准化均值）
prof_mean = prof[[c for c in prof.columns if c.endswith("_mean")]].copy()
prof_mean.columns = [c[:-5] for c in prof_mean.columns]
prof_mean.index = prof["error_type"]

# 选业务可解释的核心特征
profile_feats = [
    "user_pv_count", "item_pv_count", "item_buy_count", "item_buy_user_count",
    "buy_conversion_rate", "cart_to_buy_rate", "user_category_pref_score",
    "rfm_r_score", "rfm_f_score", "rfm_m_score",
]
profile_feats = [f for f in profile_feats if f in prof_mean.columns]

fig_prof = go.Figure()
colors_err = {"false_positive": "#ff7f0e", "false_negative": "#1f77b4"}
names_err = {"false_positive": "假阳性 FP（预测买·实际未买）",
             "false_negative": "假阴性 FN（预测未买·实际买了）"}
for et in ["false_positive", "false_negative"]:
    if et in prof_mean.index:
        fig_prof.add_trace(go.Bar(
            x=[FEATURE_CN.get(f, f) for f in profile_feats],
            y=prof_mean.loc[et, profile_feats].values,
            name=names_err[et], marker_color=colors_err[et], opacity=0.85,
            customdata=np.tile(prof_mean.loc[et, profile_feats].values, (1, 1)),
            hovertemplate="%{x}<br>标准化均值: %{y:+.2f}<extra></extra>"))
fig_prof.add_hline(y=0, line_width=1, line_color="gray")
fig_prof.update_layout(
    title="FP / FN 样本特征画像（标准化均值，0 = 全体均值水平）",
    xaxis_title="", yaxis_title="标准化均值（z-score）",
    height=440, barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
st.plotly_chart(fig_prof, use_container_width=True)

st.caption(
    "📌 FN（漏判）画像：用户活跃度、商品热度与用户购买力（RFM）均值都高于全体——"
    "模型对『高价值用户的购买』反而容易漏掉，因为这类购买常发生在交互稀疏的场景。"
    "FP（误判）画像：集中在高购买率类目 + 热门商品（类目编码/商品购买数偏高），"
    "模型把『环境看起来会买』误判为『真的会买』。"
)

# ---- 5.2 冷启动/长尾专项对比 ----
st.subheader("2️⃣ 冷启动与长尾场景专项对比")
seg_m = load_segment_metrics()
seg_name_map = {
    "overall": "整体",
    "cold_start_user_bottom20_entities": "冷启动用户<br>(活跃度 Bottom 20%)",
    "long_tail_item_bottom20_entities": "长尾商品<br>(热度 Bottom 20%)",
}
seg_m["显示名"] = seg_m["segment"].map(seg_name_map)

fig_cold = make_subplots(rows=1, cols=2, subplot_titles=("ROC-AUC", "召回率 / F1"))
x_labels = seg_m["显示名"]
fig_cold.add_trace(go.Bar(
    x=x_labels, y=seg_m["roc_auc"], marker_color="#1f77b4",
    text=[f"{v:.3f}" for v in seg_m["roc_auc"]], textposition="outside",
    customdata=seg_m["rows"], hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.4f}<br>样本数: %{customdata:,}<extra></extra>",
    showlegend=False), row=1, col=1)
fig_cold.add_trace(go.Bar(
    x=x_labels, y=seg_m["recall"], name="召回率", marker_color="#d62728",
    text=[f"{v:.1%}" for v in seg_m["recall"]], textposition="outside",
    hovertemplate="召回率: %{y:.2%}<extra></extra>"), row=1, col=2)
fig_cold.add_trace(go.Bar(
    x=x_labels, y=seg_m["f1"], name="F1", marker_color="#2ca02c",
    text=[f"{v:.3f}" for v in seg_m["f1"]], textposition="outside",
    hovertemplate="F1: %{y:.3f}<extra></extra>"), row=1, col=2)
fig_cold.update_layout(height=420, barmode="group",
                       legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
fig_cold.update_yaxes(range=[0, 1.12], row=1, col=1)
fig_cold.update_yaxes(range=[0, 1.12], tickformat=".0%", row=1, col=2)
st.plotly_chart(fig_cold, use_container_width=True)

disp_seg_m = seg_m[["segment", "rows", "positive_count", "roc_auc", "pr_auc_ap",
                    "precision", "recall", "f1"]].copy()
disp_seg_m["segment"] = disp_seg_m["segment"].map(
    lambda s: seg_name_map.get(s, s).replace("<br>", " "))
for c_ in ["roc_auc", "pr_auc_ap", "precision", "recall", "f1"]:
    disp_seg_m[c_] = disp_seg_m[c_].map(lambda v: f"{v:.4f}")
st.dataframe(disp_seg_m, use_container_width=True, hide_index=True)

st.caption(
    "📌 冷启动用户场景 F1 仅 0.085（整体 0.192），召回率 15.2%（整体 51.2%）——"
    "交互稀疏导致行为特征失效，是明确的模型短板；长尾商品场景反而保持 0.363 的 F1。"
    "优化方向：新用户冷启动特征（注册时长/首日行为）+ 基于内容/类目的相似度召回。"
)

# ---- 5.3 高置信误判明细 ----
st.subheader("3️⃣ 高置信误判样本明细（Top 错误样本）")
top_err = load_top_errors()
err_filter = st.radio(
    "错误类型", ["false_positive", "false_negative"], horizontal=True,
    format_func=lambda x: "假阳性 FP（预测买·实际未买）" if x == "false_positive"
    else "假阴性 FN（预测未买·实际买了）", key="err_type_filter")
show_err = top_err[top_err["error_type"] == err_filter].head(20).copy()
show_err["purchase_probability"] = show_err["purchase_probability"].map(lambda v: f"{v:.3%}")

disp_cols = [c for c in ["user_id", "item_id", "y_true", "purchase_probability",
                         "user_pv_count", "item_pv_count", "item_buy_count",
                         "buy_conversion_rate", "rfm_r_score", "rfm_f_score",
                         "rfm_m_score", "user_category_pref_score"] if c in show_err.columns]
st.dataframe(show_err[disp_cols], use_container_width=True, hide_index=True)
st.caption(BEHAVIOR_NOTE + "；明细按预测置信度排序，用于逐案归因。")

# ---- 页脚 ----
st.markdown("---")
st.caption(
    "数据来源：output/step9_test_model_evaluation（测试集评估）、"
    "output/step9_test_interpretability_error_analysis（SHAP 与错误分析）、"
    "output/baseline_models、output/optuna_tuned_models、output/tree_stacking、"
    "output/part7_din_stacking（多模型对比）。评估口径详见 docs/Part6、Part7、step9 文档。"
)
