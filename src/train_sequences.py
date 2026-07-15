"""Part7 LSTM/GRU 序列模型训练 —— 主入口。

已完成的三部曲基线模型（LR/LightGBM/XGBoost）使用聚合快照特征，
忽略了用户-商品交互的时间顺序信息。
本模块从原始行为日志 rebuilt 31 天序列，训练 LSTM/GRU 捕捉时序模式。

前置条件：
    train/val/test.parquet 必须包含 item_id 列。
    如果当前缺少，重新运行：
        python src/build_samples.py

运行方式:
    cd "/Users/yangzhuoyao/Desktop/阿里/项目1"
    python src/train_sequences.py

数据流：
    cleaned_data.parquet (12M 行, 有 time)
    → builder: 按 (user_id, item_id, date) 日聚合
    → builder: 匹配 train/val/test 的 (user_id, item_id)
    → output/sequences/train_sequences.npy (N_train, 31, 7)
    → dataset: BehaviorSequenceDataset
    → trainer: LSTM / GRU 训练 + 评估
"""

from __future__ import annotations

import os
import sys

# 切换工作目录到项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.logger import get_logger
from sequence_building.builder import build_all_sequences
from sequence_building.dataset import BehaviorSequenceDataset
from sequence_building.trainer import DEVICE, train_gru, train_lstm

logger = get_logger(__name__)

# ---- 路径常量 ----
CLEANED_DATA_PATH = "output/cleaned_data.parquet"
TRAIN_PATH = "output/train.parquet"
VAL_PATH = "output/val.parquet"
TEST_PATH = "output/test.parquet"
SEQ_OUTPUT_DIR = "output/sequences"
MODEL_OUTPUT_DIR = "output/models"

# ---- 训练超参数 ----
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 10


def _compute_pos_weight(train_dataset: BehaviorSequenceDataset) -> float:
    """根据训练集正负样本比例计算 pos_weight。

    pos_weight = 负样本数 / 正样本数，用于 BCEWithLogitsLoss
    告诉模型：少数类（正样本=购买）的重要性是负样本的 N 倍。

    Args:
        train_dataset: 训练数据集

    Returns:
        pos_weight 值
    """
    labels = train_dataset.labels.numpy()
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _create_dataloaders(
    train_seq_path: str,
    train_label_path: str,
    val_seq_path: str,
    val_label_path: str,
    test_seq_path: str,
    test_label_path: str,
) -> dict:
    """创建训练/验证/测试 DataLoader。

    Returns:
        {
            "train_loader": ...,
            "val_loader": ...,
            "test_loader": ...,
            "train_dataset": ...,
            "val_dataset": ...,
            "test_dataset": ...,
            "pos_weight": ...,
        }
    """
    logger.info("创建 DataLoaders (batch_size=%d) ...", BATCH_SIZE)

    train_dataset = BehaviorSequenceDataset(
        train_seq_path, train_label_path
    )
    val_dataset = BehaviorSequenceDataset(val_seq_path, val_label_path)
    test_dataset = BehaviorSequenceDataset(
        test_seq_path, test_label_path
    )

    pos_weight = _compute_pos_weight(train_dataset)

    logger.info("训练集: %s", train_dataset.summary())
    logger.info("验证集: %s", val_dataset.summary())
    logger.info("测试集: %s", test_dataset.summary())
    logger.info("pos_weight: %.2f", pos_weight)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "pos_weight": pos_weight,
    }


def main() -> None:
    """主流程：构建序列 → 创建 DataLoader → 训练 LSTM → 训练 GRU → 汇总对比。"""
    # ---- 前置检查 ----
    logger.info("=" * 60)
    logger.info("Part 7: LSTM/GRU 序列模型训练")
    logger.info("=" * 60)

    # 检查 train.parquet 是否有 item_id
    import pandas as pd

    df_train_check = pd.read_parquet(TRAIN_PATH)
    if "item_id" not in df_train_check.columns:
        logger.error(
            "train.parquet 缺少 item_id 列！"
        )
        logger.error(
            "请先重新运行 build_samples 来生成带 item_id 的样本："
        )
        logger.error("    python src/build_samples.py")
        return

    logger.info("train.parquet 已包含 item_id 列，继续 ...")

    # ---- Step 1: 构建序列 ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 1/3: 构建 31 天行为序列")
    logger.info("=" * 60)

    # 检查序列是否已存在，跳过重建
    train_seq_path = os.path.join(SEQ_OUTPUT_DIR, "train_sequences.npy")
    if os.path.exists(train_seq_path):
        logger.info("序列文件已存在，跳过构建")
        logger.info("如需重建，请删除 %s 后重新运行", SEQ_OUTPUT_DIR)
        seq_paths = {
            "train_seq": train_seq_path,
            "train_label": os.path.join(SEQ_OUTPUT_DIR, "train_labels.npy"),
            "val_seq": os.path.join(SEQ_OUTPUT_DIR, "val_sequences.npy"),
            "val_label": os.path.join(SEQ_OUTPUT_DIR, "val_labels.npy"),
            "test_seq": os.path.join(SEQ_OUTPUT_DIR, "test_sequences.npy"),
            "test_label": os.path.join(SEQ_OUTPUT_DIR, "test_labels.npy"),
        }
    else:
        seq_paths = build_all_sequences(
            cleaned_data_path=CLEANED_DATA_PATH,
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_path=TEST_PATH,
            output_dir=SEQ_OUTPUT_DIR,
        )

    # ---- Step 2: 创建 DataLoader ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 2/3: 创建 DataLoaders")
    logger.info("=" * 60)

    dataloaders = _create_dataloaders(
        train_seq_path=seq_paths["train_seq"],
        train_label_path=seq_paths["train_label"],
        val_seq_path=seq_paths["val_seq"],
        val_label_path=seq_paths["val_label"],
        test_seq_path=seq_paths["test_seq"],
        test_label_path=seq_paths["test_label"],
    )

    # ---- Step 3: 训练 LSTM + GRU ----
    logger.info("\n" + "=" * 60)
    logger.info("Step 3/3: 训练 LSTM + GRU")
    logger.info("=" * 60)

    results = {}

    logger.info("\n>>> 训练 LSTM ...")
    results["lstm"] = train_lstm(
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        output_dir=MODEL_OUTPUT_DIR,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pos_weight=dataloaders["pos_weight"],
        patience=PATIENCE,
    )

    logger.info("\n>>> 训练 GRU ...")
    results["gru"] = train_gru(
        train_loader=dataloaders["train_loader"],
        val_loader=dataloaders["val_loader"],
        output_dir=MODEL_OUTPUT_DIR,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        pos_weight=dataloaders["pos_weight"],
        patience=PATIENCE,
    )

    # ---- 汇总对比 ----
    logger.info("\n" + "=" * 60)
    logger.info("序列模型性能汇总")
    logger.info("=" * 60)
    logger.info(
        "%-12s | %8s | %8s | %8s | %8s | %8s | %8s"
        % ("模型", "Epoch", "Train F1", "Val F1", "Val AUC", "Precision", "Recall")
    )
    logger.info("-" * 72)
    for name in ["lstm", "gru"]:
        r = results[name]
        vm = r["val_metrics"]
        tm = r["train_metrics"]
        logger.info(
            "%-12s | %8d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f",
            name.upper(),
            r["best_epoch"],
            tm["f1"],
            vm["f1"],
            vm["auc"],
            vm["precision"],
            vm["recall"],
        )

    logger.info("\n模型保存位置: %s", MODEL_OUTPUT_DIR)
    logger.info("序列文件位置: %s", SEQ_OUTPUT_DIR)
    logger.info("\nPart 7 完成！")


if __name__ == "__main__":
    main()
