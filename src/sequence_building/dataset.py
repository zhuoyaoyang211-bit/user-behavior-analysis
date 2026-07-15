"""PyTorch Dataset 封装。

将 builder.py 输出的 numpy 序列数据封装为 PyTorch Dataset，
支持 DataLoader 的批处理训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BehaviorSequenceDataset(Dataset):
    """用户-商品行为序列数据集。

    每个样本为 (sequence, label)：
    - sequence: (31, 7) float32，31 天 × 7 特征
    - label: int，0=未购买，1=购买
    """

    def __init__(
        self,
        seq_path: str | Path,
        label_path: str | Path,
        device: str = "cpu",
    ) -> None:
        """加载序列数据和标签。

        使用 memmap（内存映射）模式读取序列：
        - 2.85GB 的序列文件留在硬盘上，不在内存中占位置
        - DataLoader 取 batch 时才读取对应行到内存
        - 注意：要求 DataLoader 使用 num_workers=0（单进程），
          否则多进程会尝试共享同一个 memmap 导致冲突

        Args:
            seq_path: .npy 序列文件路径，shape (N, 31, 7)
            label_path: .npy 标签文件路径，shape (N,)
            device: 数据加载到的设备 ("cpu" / "cuda" / "mps")
        """
        self.sequences = torch.from_numpy(
            np.load(seq_path, mmap_mode="r")
        )
        self.labels = torch.from_numpy(
            np.load(label_path).astype(np.int64)
        )
        self.device = device

        # 验证数据一致性
        assert len(self.sequences) == len(self.labels), (
            f"序列数 ({len(self.sequences)}) 与标签数 ({len(self.labels)}) 不一致"
        )
        assert self.sequences.shape[1:] == (31, 7), (
            f"序列 shape 应为 (N, 31, 7)，实际为 {self.sequences.shape}"
        )

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            (sequence, label) 元组：
            - sequence: (31, 7) float32
            - label: 标量 int
        """
        return self.sequences[idx], self.labels[idx]

    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回原始 numpy 数组，用于非 PyTorch 场景。

        Returns:
            (sequences, labels)
        """
        return self.sequences.numpy(), self.labels.numpy()

    @property
    def pos_ratio(self) -> float:
        """正样本占比。"""
        return float(self.labels.float().mean())

    @property
    def n_features(self) -> int:
        """特征维度（=7）。"""
        return self.sequences.shape[2]

    @property
    def seq_len(self) -> int:
        """序列长度（=31 天）。"""
        return self.sequences.shape[1]

    def summary(self) -> str:
        """返回数据集摘要信息。"""
        n = len(self)
        pos = int(self.labels.sum())
        return (
            f"样本: {n:,}  |  正样本: {pos:,} ({pos/n*100:.2f}%)  "
            f"|  负样本: {n-pos:,} ({(n-pos)/n*100:.2f}%)  "
            f"|  Shape: {self.sequences.shape}"
        )
