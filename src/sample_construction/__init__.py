"""Part5 样本构建模块。

包含样本划分、不平衡处理、三种方案对比评估。
"""

from sample_construction.builder import build_samples
from sample_construction.imbalance import (
    apply_class_weight,
    apply_smote,
    apply_undersample,
)
from sample_construction.compare import compare_methods

__all__ = [
    "build_samples",
    "apply_smote",
    "apply_undersample",
    "apply_class_weight",
    "compare_methods",
]
