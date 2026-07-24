"""Neural sequence models for purchase prediction."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


RnnCellType = Literal["lstm", "gru"]


class SequenceEmbedding(nn.Module):
    """Shared item, category, behavior, and time-gap event embeddings."""

    def __init__(
        self,
        item_vocab_size: int,
        category_vocab_size: int,
        behavior_vocab_size: int,
        embedding_dim: int,
        behavior_embedding_dim: int,
    ) -> None:
        """Initialize embedding tables.

        Args:
            item_vocab_size: Number of item embedding rows including padding.
            category_vocab_size: Number of category rows including padding.
            behavior_vocab_size: Number of behavior rows including padding.
            embedding_dim: Item and category embedding dimension.
            behavior_embedding_dim: Behavior embedding dimension.
        """
        super().__init__()
        self.item_embedding = nn.Embedding(
            item_vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        self.category_embedding = nn.Embedding(
            category_vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        self.behavior_embedding = nn.Embedding(
            behavior_vocab_size,
            behavior_embedding_dim,
            padding_idx=0,
        )
        self.event_dim = embedding_dim * 2 + behavior_embedding_dim + 1
        self.target_dim = embedding_dim * 2

    def embed_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed historical events and concatenate normalized time gaps."""
        item_embedding = self.item_embedding(batch["hist_item"])
        category_embedding = self.category_embedding(batch["hist_category"])
        behavior_embedding = self.behavior_embedding(batch["hist_behavior"])
        time_gap = batch["hist_time_gap"].unsqueeze(-1)
        return torch.cat(
            [item_embedding, category_embedding, behavior_embedding, time_gap],
            dim=-1,
        )

    def embed_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed target item and target category."""
        item_embedding = self.item_embedding(batch["target_item"])
        category_embedding = self.category_embedding(batch["target_category"])
        return torch.cat([item_embedding, category_embedding], dim=-1)


class RnnSequenceClassifier(nn.Module):
    """LSTM or GRU sequence classifier baseline."""

    def __init__(
        self,
        cell_type: RnnCellType,
        item_vocab_size: int,
        category_vocab_size: int,
        behavior_vocab_size: int,
        embedding_dim: int,
        behavior_embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize an RNN sequence classifier.

        Args:
            cell_type: "lstm" or "gru".
            item_vocab_size: Number of item embedding rows.
            category_vocab_size: Number of category embedding rows.
            behavior_vocab_size: Number of behavior embedding rows.
            embedding_dim: Item/category embedding dimension.
            behavior_embedding_dim: Behavior embedding dimension.
            hidden_size: Recurrent hidden size.
            num_layers: Number of recurrent layers.
            dropout: Dropout rate for the prediction head.
        """
        super().__init__()
        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"Unsupported RNN cell type: {cell_type}")

        self.cell_type = cell_type
        self.embedding = SequenceEmbedding(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            behavior_vocab_size=behavior_vocab_size,
            embedding_dim=embedding_dim,
            behavior_embedding_dim=behavior_embedding_dim,
        )
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=self.embedding.event_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + self.embedding.target_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return purchase logits for one mini-batch."""
        history_embedding = self.embedding.embed_history(batch)
        lengths = batch["seq_len"]
        packed_lengths = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            history_embedding,
            packed_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)
        if self.cell_type == "lstm":
            hidden_state = hidden[0][-1]
        else:
            hidden_state = hidden[-1]

        has_history = lengths.gt(0).float().unsqueeze(1)
        user_interest = hidden_state * has_history
        target_embedding = self.embedding.embed_target(batch)
        logits = self.head(torch.cat([user_interest, target_embedding], dim=1))
        return logits.squeeze(1)


class DinClassifier(nn.Module):
    """Deep Interest Network classifier for target-aware user interest."""

    def __init__(
        self,
        item_vocab_size: int,
        category_vocab_size: int,
        behavior_vocab_size: int,
        embedding_dim: int,
        behavior_embedding_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        """Initialize DIN classifier.

        Args:
            item_vocab_size: Number of item embedding rows.
            category_vocab_size: Number of category embedding rows.
            behavior_vocab_size: Number of behavior embedding rows.
            embedding_dim: Item/category embedding dimension.
            behavior_embedding_dim: Behavior embedding dimension.
            hidden_size: Projection size for attention representations.
            dropout: Dropout rate for attention and prediction MLPs.
        """
        super().__init__()
        self.embedding = SequenceEmbedding(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            behavior_vocab_size=behavior_vocab_size,
            embedding_dim=embedding_dim,
            behavior_embedding_dim=behavior_embedding_dim,
        )
        self.history_projection = nn.Linear(self.embedding.event_dim, hidden_size)
        self.target_projection = nn.Linear(self.embedding.target_dim, hidden_size)
        attention_input_dim = hidden_size * 4
        self.attention = nn.Sequential(
            nn.Linear(attention_input_dim, 64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return purchase logits for one mini-batch."""
        history_embedding = self.embedding.embed_history(batch)
        target_embedding = self.embedding.embed_target(batch)
        history_repr = self.history_projection(history_embedding)
        target_repr = self.target_projection(target_embedding)

        sequence_length = history_repr.size(1)
        expanded_target = target_repr.unsqueeze(1).expand(-1, sequence_length, -1)
        attention_input = torch.cat(
            [
                history_repr,
                expanded_target,
                history_repr - expanded_target,
                history_repr * expanded_target,
            ],
            dim=-1,
        )
        attention_score = self.attention(attention_input).squeeze(-1)
        steps = torch.arange(sequence_length, device=attention_score.device)
        mask = steps.unsqueeze(0) < batch["seq_len"].unsqueeze(1)
        attention_score = attention_score.masked_fill(~mask, -1e9)
        attention_weight = torch.softmax(attention_score, dim=1) * mask.float()
        attention_weight = attention_weight / attention_weight.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
        user_interest = torch.bmm(
            attention_weight.unsqueeze(1),
            history_repr,
        ).squeeze(1)

        logits = self.head(torch.cat([user_interest, target_repr], dim=1))
        return logits.squeeze(1)


def build_sequence_model(
    model_name: Literal["lstm", "gru", "din"],
    item_vocab_size: int,
    category_vocab_size: int,
    behavior_vocab_size: int,
    embedding_dim: int,
    behavior_embedding_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a sequence model by name.

    Args:
        model_name: Sequence model name.
        item_vocab_size: Number of item embedding rows.
        category_vocab_size: Number of category embedding rows.
        behavior_vocab_size: Number of behavior embedding rows.
        embedding_dim: Item/category embedding dimension.
        behavior_embedding_dim: Behavior embedding dimension.
        hidden_size: RNN hidden size or DIN projection size.
        num_layers: Number of recurrent layers for RNN models.
        dropout: Dropout rate.

    Returns:
        Initialized PyTorch module.
    """
    if model_name in ("lstm", "gru"):
        return RnnSequenceClassifier(
            cell_type=model_name,
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            behavior_vocab_size=behavior_vocab_size,
            embedding_dim=embedding_dim,
            behavior_embedding_dim=behavior_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_name == "din":
        return DinClassifier(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            behavior_vocab_size=behavior_vocab_size,
            embedding_dim=embedding_dim,
            behavior_embedding_dim=behavior_embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model name: {model_name}")
