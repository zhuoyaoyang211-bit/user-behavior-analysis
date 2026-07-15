"""LSTM/GRU 序列模型定义与训练。

包含两个模型类（LSTMClassifier、GRUClassifier）和统一的训练/评估流程。
支持类别不平衡处理（pos_weight）、早停、学习率调度。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from common.logger import get_logger

logger = get_logger(__name__)

# ---- 设备检测 ----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

logger.info("使用设备: %s", DEVICE)


# ============================================================
# 模型定义
# ============================================================


class LSTMClassifier(nn.Module):
    """双向 LSTM 二分类器。

    结构：
        Input (N, 31, 7)
        → BiLSTM (hidden=64, layers=2)
        → Concat last hidden states (forward + backward)
        → Dropout
        → Linear → 1 (logit)

    Args:
        input_size: 输入特征维度（默认 7）
        hidden_size: LSTM 隐藏层维度（默认 64）
        num_layers: LSTM 层数（默认 2）
        dropout: Dropout 比率（默认 0.3）
        bidirectional: 是否双向（默认 True）
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 分类头：拼接双向最后一层 hidden → logit
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, 1)
        """
        # LSTM 输出: output (batch, seq, hidden*D), (h_n, c_n)
        output, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # 取最后两层（forward 和 backward）的最后时刻 hidden
            h_forward = h_n[-2, :, :]  # 正向最后一层
            h_backward = h_n[-1, :, :]  # 反向最后一层
            h_cat = torch.cat([h_forward, h_backward], dim=1)  # (batch, H*2)
        else:
            h_cat = h_n[-1, :, :]  # (batch, H)

        out = self.dropout(h_cat)
        out = self.fc(out)  # (batch, 1)
        return out


class GRUClassifier(nn.Module):
    """双向 GRU 二分类器。

    结构与 LSTMClassifier 一致，仅将 LSTM 替换为 GRU。

    Args:
        input_size: 输入特征维度（默认 7）
        hidden_size: GRU 隐藏层维度（默认 64）
        num_layers: GRU 层数（默认 2）
        dropout: Dropout 比率（默认 0.3）
        bidirectional: 是否双向（默认 True）
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, 1)
        """
        # GRU 不返回细胞状态，h_n shape: (D*num_layers, batch, hidden)
        output, h_n = self.gru(x)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_cat = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_cat = h_n[-1, :, :]

        out = self.dropout(h_cat)
        out = self.fc(out)
        return out


# ============================================================
# 评估函数
# ============================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """在给定 DataLoader 上评估模型。

    Args:
        model: 待评估模型
        loader: 数据加载器
        device: 计算设备

    Returns:
        {
            "loss": 平均损失,
            "accuracy": 准确率,
            "precision": 精确率,
            "recall": 召回率,
            "f1": F1 分数,
            "auc": AUC (正样本需 ≥ 2 类),
        }
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)

        logits = model(batch_x).squeeze(-1)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * len(batch_x)

        probs = torch.sigmoid(logits)
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
    }

    # AUC: 需要至少一类各一个样本
    if len(np.unique(all_labels)) >= 2:
        metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
    else:
        metrics["auc"] = 0.5

    return metrics


# ============================================================
# 训练函数
# ============================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    output_dir: str | Path,
    *,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    pos_weight: float | None = None,
    patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """训练并评估一个序列模型。

    训练策略：
        - 损失函数: BCEWithLogitsLoss（支持 pos_weight 处理样本不平衡）
        - 优化器: Adam
        - 学习率调度: ReduceLROnPlateau（val_loss 不降时减半）
        - 早停: 监控 val_f1，patience 个 epoch 不提升则停止
        - 模型保存: 保留验证集 F1 最高的模型

    Args:
        model: 待训练的模型实例
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        model_name: 模型名称（用于日志和保存）
        output_dir: 模型保存目录
        epochs: 最大训练轮数
        lr: 初始学习率
        weight_decay: L2 正则化系数
        pos_weight: 正样本权重（= 负样本数 / 正样本数），自动计算
        patience: 早停容忍轮数
        scheduler_factor: 学习率衰减因子
        scheduler_patience: 学习率调度容忍轮数
        device: 计算设备

    Returns:
        {
            "model_name": 模型名称,
            "train_time": 训练耗时(秒),
            "best_epoch": 最佳轮次,
            "best_val_f1": 最佳验证集 F1,
            "best_val_auc": 最佳验证集 AUC,
            "train_metrics": 训练集指标,
            "val_metrics": 验证集指标,
            "model_path": 保存的模型路径,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # 类别权重（处理样本不平衡）
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=device)
    else:
        pw = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=False,
    )

    best_val_f1 = 0.0
    best_epoch = 0
    best_state_dict = None
    no_improve = 0

    logger.info("=" * 60)
    logger.info("开始训练 %s", model_name)
    logger.info("  设备: %s, 学习率: %.4f, Epochs: %d", device, lr, epochs)
    logger.info("  早停: %d epochs, pos_weight=%s", patience, f"{pos_weight:.1f}" if pos_weight else "无")
    logger.info("=" * 60)

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_x)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # ---- 验证 ----
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %2d | Train Loss: %.4f | Val Loss: %.4f | "
            "Val Acc: %.4f | Val F1: %.4f | Val AUC: %.4f | LR: %.6f",
            epoch,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            val_metrics["auc"],
            current_lr,
        )

        # ---- 早停 & 保存最佳模型 ----
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            best_state_dict = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    "早停触发: 连续 %d 个 epoch F1 未提升", patience
                )
                break

    train_time = time.time() - t_start

    # 恢复最佳模型
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 最终评估
    logger.info("计算最终训练集指标 ...")
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    # 保存模型
    model_path = str(output_dir / f"{model_name}.pth")
    torch.save(
        {
            "model_state_dict": best_state_dict if best_state_dict else model.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
        model_path,
    )

    logger.info("-" * 40)
    logger.info("%s 训练完成（耗时 %.1f 秒）", model_name, train_time)
    logger.info(
        "  最佳 Epoch: %d  |  Val F1: %.4f  |  Val AUC: %.4f",
        best_epoch, best_val_f1, val_metrics["auc"],
    )
    logger.info(
        "  Train: Loss=%.4f Acc=%.4f F1=%.4f AUC=%.4f",
        train_metrics["loss"], train_metrics["accuracy"],
        train_metrics["f1"], train_metrics["auc"],
    )
    logger.info(
        "  Val:   Loss=%.4f Acc=%.4f F1=%.4f AUC=%.4f",
        val_metrics["loss"], val_metrics["accuracy"],
        val_metrics["f1"], val_metrics["auc"],
    )
    logger.info("  模型已保存: %s", model_path)

    return {
        "model_name": model_name,
        "train_time": round(train_time, 1),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_auc": val_metrics["auc"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": model_path,
    }


# ============================================================
# 便捷入口
# ============================================================


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    **kwargs,
) -> Dict[str, Any]:
    """训练 LSTM 模型。

    参数同 train_model()，默认配置见下方。
    """
    model = LSTMClassifier(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LSTM 参数量: 总计 %s, 可训练 %s", f"{total_params:,}", f"{trainable_params:,}")

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="lstm_baseline",
        output_dir=output_dir,
        **kwargs,
    )


def train_gru(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    **kwargs,
) -> Dict[str, Any]:
    """训练 GRU 模型。

    参数同 train_model()，默认配置见下方。
    """
    model = GRUClassifier(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("GRU 参数量: 总计 %s, 可训练 %s", f"{total_params:,}", f"{trainable_params:,}")

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="gru_baseline",
        output_dir=output_dir,
        **kwargs,
    )
