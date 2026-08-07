# Step9 模型可解释性与错误样本分析报告

## 1. 分析配置

- 测试集：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/test.parquet`
- 最终模型：优化后的 XGBoost，`/Users/yangzhuoyao/Desktop/阿里/项目1/output/optuna_tuned_models/xgboost_optuna.joblib`
- 预测结果：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/step9_test_model_evaluation/step9_test_predictions.csv`
- 输出目录：`/Users/yangzhuoyao/Desktop/阿里/项目1/output/step9_test_interpretability_error_analysis`
- 样本数：468,316
- 正样本占比：0.095875%
- 固定分类阈值：0.2
- 阈值来源：`internal_tuning_validation`
- XGBoost SHAP 分析样本数：10,000
- SHAP 样本正样本数：449，负样本数：9,551，正样本占比：4.490000%

## 2. XGBoost 特征重要性

树模型内置特征重要性反映特征在树分裂中的贡献频率或增益综合表现，可用于观察模型主要依赖哪些业务特征。

| feature | feature_importance |
| --- | --- |
| item_buy_count | 0.486388 |
| item_buy_user_count | 0.194797 |
| item_pv_to_buy_rate | 0.114317 |
| user_category_pref_score | 0.040341 |
| item_pv_count | 0.027037 |
| item_cart_to_buy_rate | 0.024865 |
| item_fav_count | 0.018571 |
| rfm_f_score | 0.015322 |
| item_cart_count | 0.012256 |
| repurchase_item_count | 0.010752 |

## 3. XGBoost 原生 SHAP 解释

本部分使用 XGBoost 原生 `pred_contribs=True` 计算 SHAP contribution values。`mean_abs_shap` 越高，说明该特征对模型输出的影响越大。由于购买正样本极少，全体抽样的 `mean_shap` 会被负样本主导，因此不直接把它解释为特征的业务方向。本报告改用正负标签组的平均 SHAP 贡献及其差值，描述该特征在两类样本中的输出贡献差异。SHAP 抽样保留测试集全部正样本并抽取负样本，目的是保证稀有正样本可解释，因此该表用于机制分析，不作为测试集总体分布的重新估计。

| feature | mean_abs_shap | positive_label_mean_shap | negative_label_mean_shap | positive_minus_negative_mean_shap | contrast_direction |
| --- | --- | --- | --- | --- | --- |
| item_buy_count | 3.488111 | 1.627275 | -3.392395 | 5.019670 | more_positive_for_positive_labels |
| user_category_pref_score | 2.511658 | 0.151199 | -2.570621 | 2.721821 | more_positive_for_positive_labels |
| item_buy_user_count | 1.596576 | 0.615686 | -1.577097 | 2.192783 | more_positive_for_positive_labels |
| item_pv_to_buy_rate | 1.170681 | 0.479613 | -1.155816 | 1.635429 | more_positive_for_positive_labels |
| buy_conversion_rate | 0.409266 | 0.014235 | -0.376837 | 0.391072 | more_positive_for_positive_labels |
| repurchase_item_count | 0.375609 | -0.388677 | -0.370983 | -0.017694 | more_negative_for_positive_labels |
| item_pv_count | 0.366823 | -0.219297 | -0.338091 | 0.118795 | more_positive_for_positive_labels |
| cat_view_user_count | 0.306133 | -0.090100 | -0.246537 | 0.156438 | more_positive_for_positive_labels |
| item_cart_to_buy_rate | 0.259082 | 0.193766 | -0.244883 | 0.438649 | more_positive_for_positive_labels |
| item_category_te | 0.201081 | -0.099171 | -0.190194 | 0.091023 | more_positive_for_positive_labels |

## 4. 冷启动与长尾场景分段表现

冷启动用户和长尾商品分别按照测试集中用户实体、商品实体的行为特征排序，选取底部 20% 的实体进行诊断。由于每个实体包含的用户-商品样本数不同，对应的行占比不一定是 20%。部分特征已经标准化，因此阈值只用于相对分组，不直接解释为原始浏览次数。这些分段不是重新训练样本，只是测试集上的诊断切片。

| segment | rows | positive_count | positive_rate | accuracy | precision | recall | f1 | tn | fp | fn | tp | roc_auc | pr_auc_ap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 468316 | 449 | 0.000959 | 0.995866 | 0.118130 | 0.512249 | 0.191987 | 466150 | 1717 | 219 | 230 | 0.994194 | 0.131205 |
| cold_start_user_bottom20_entities | 15145 | 33 | 0.002179 | 0.992935 | 0.059524 | 0.151515 | 0.085470 | 15033 | 79 | 28 | 5 | 0.982117 | 0.069190 |
| long_tail_item_bottom20_entities | 80876 | 89 | 0.001100 | 0.997131 | 0.240000 | 0.741573 | 0.362637 | 80578 | 209 | 23 | 66 | 0.998251 | 0.252271 |

## 5. 错误样本特征画像

下表按 false positive、false negative、true positive、true negative 聚合关键特征均值和中位数，用于定位误判样本的共性。特征数值为标准化后的相对值，不代表原始业务计数。

| error_type | user_pv_count_count | user_pv_count_mean | user_pv_count_median | item_pv_count_count | item_pv_count_mean | item_pv_count_median | item_buy_count_count | item_buy_count_mean | item_buy_count_median | item_buy_user_count_count | item_buy_user_count_mean | item_buy_user_count_median | cat_pv_count_count | cat_pv_count_mean | cat_pv_count_median | cat_view_user_count_count | cat_view_user_count_mean | cat_view_user_count_median | buy_conversion_rate_count | buy_conversion_rate_mean | buy_conversion_rate_median | cart_to_buy_rate_count | cart_to_buy_rate_mean | cart_to_buy_rate_median | fav_to_buy_rate_count | fav_to_buy_rate_mean | fav_to_buy_rate_median | user_category_pref_score_count | user_category_pref_score_mean | user_category_pref_score_median | user_avg_interval_hours_count | user_avg_interval_hours_mean | user_avg_interval_hours_median | item_decay_slope_count | item_decay_slope_mean | item_decay_slope_median | rfm_r_score_count | rfm_r_score_mean | rfm_r_score_median | rfm_f_score_count | rfm_f_score_mean | rfm_f_score_median | rfm_m_score_count | rfm_m_score_mean | rfm_m_score_median | purchase_probability_count | purchase_probability_mean | purchase_probability_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| false_negative | 219 | -0.211005 | -0.492690 | 219 | 0.029819 | -0.206122 | 219 | 1.572157 | 0.855667 | 219 | 1.625622 | 1.151014 | 219 | -0.346767 | -0.655886 | 219 | -0.398696 | -0.748157 | 219 | 1.454602 | 0.661798 | 219 | 0.534757 | 0.259120 | 219 | 0.235365 | -0.264626 | 219 | 1.077431 | 0.530041 | 219 | 0.346373 | -0.216299 | 219 | 0.072000 | 0.039579 | 219 | 0.191704 | 0.398225 | 219 | 0.499431 | 1.021063 | 219 | 0.449038 | 0.730710 | 219 | 0.082934 | 0.072386 |
| false_positive | 1717 | -0.253285 | -0.396494 | 1717 | -0.209653 | -0.287783 | 1717 | 1.110815 | 0.855667 | 1717 | 1.266509 | 1.151014 | 1717 | -0.569880 | -0.687932 | 1717 | -0.713661 | -0.930401 | 1717 | 1.593629 | 0.937042 | 1717 | 0.346052 | 0.174758 | 1717 | 0.120010 | -0.274020 | 1717 | 0.935168 | 0.286912 | 1717 | -0.005823 | -0.234121 | 1717 | 0.185210 | 0.039579 | 1717 | 0.322064 | 0.398225 | 1717 | 0.709135 | 1.021063 | 1717 | 0.580267 | 0.730710 | 1717 | 0.416590 | 0.355914 |
| true_negative | 466150 | 0.017516 | -0.279995 | 466150 | -0.010744 | -0.287783 | 466150 | -0.016227 | -0.197428 | 466150 | -0.018331 | -0.216604 | 466150 | -0.008064 | -0.499125 | 466150 | -0.008198 | -0.264724 | 466150 | -0.009748 | -0.294640 | 466150 | -0.015768 | -0.284420 | 466150 | -0.005392 | -0.398905 | 466150 | -0.017192 | -0.345224 | 466150 | 0.012129 | -0.263679 | 466150 | 0.002819 | 0.039579 | 466150 | -0.008763 | 0.398225 | 466150 | 0.003526 | 0.279262 | 466150 | -0.001719 | 0.730710 | 466150 | 0.000933 | 0.000001 |
| true_positive | 230 | -0.138820 | -0.236058 | 230 | -0.240118 | -0.308198 | 230 | 1.304378 | 0.855667 | 230 | 1.448323 | 1.151014 | 230 | -0.608042 | -0.703046 | 230 | -0.814334 | -0.997275 | 230 | 1.611957 | 1.025980 | 230 | 0.298919 | 0.165446 | 230 | -0.005877 | -0.366279 | 230 | 0.471899 | 0.128878 | 230 | -0.216972 | -0.381049 | 230 | 0.219935 | 0.039579 | 230 | 0.317002 | 0.398225 | 230 | 0.824324 | 1.021063 | 230 | 0.650250 | 0.730710 | 230 | 0.497116 | 0.443756 |

## 6. 结论口径

最终模型的主要解释对象是优化后的 XGBoost。若高影响特征集中在商品热度、用户行为强度、类目偏好和转化率特征，说明模型主要通过历史行为强度与商品受欢迎程度判断购买概率。

错误样本分析重点关注 false negative 和 false positive：false negative 代表真实购买但模型没有召回的样本，通常会暴露冷启动、低历史行为、长尾商品或类目偏好不足的问题；false positive 代表模型高估购买概率的样本，通常和高浏览、高加购或热门商品但最终未购买有关。

## 7. 产出物

| 文件 | 说明 |
| --- | --- |
| `xgboost_feature_importance.csv` | XGBoost 内置特征重要性 |
| `xgboost_native_shap_importance.csv` | XGBoost 原生 SHAP 全局重要性 |
| `xgboost_native_shap_values_sample.csv` | SHAP 样本明细 |
| `error_segment_metrics.csv` | 冷启动、长尾等分段指标 |
| `error_feature_profile.csv` | 错误类型特征画像 |
| `top_error_samples.csv` | 高置信误判样本 Top 明细 |
| `test_error_labels.csv` | 测试集逐样本错误类型 |
| `step9_interpretability_error_report.md` | 本报告 |
