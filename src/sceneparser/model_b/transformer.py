"""
Model B: Conditional Architect Transformer
==========================================
Autoregressive 4-layer Transformer architecture conditioned on a 27-dimensional
spatial semantic blueprint vector to generate 16x16 level grids.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """
    2D row-column positional embeddings for 16x16 grids.
    """

    def __init__(self, d_model: int, max_height: int = 16, max_width: int = 16) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_height, d_model // 2)
        self.col_embed = nn.Embedding(max_width, d_model // 2)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(seq_len, device=device) // 16
        cols = torch.arange(seq_len, device=device) % 16
        row_pos = self.row_embed(rows)
        col_pos = self.col_embed(cols)
        return torch.cat([row_pos, col_pos], dim=-1).unsqueeze(0)


class ConditionalArchitectTransformer(nn.Module):
    """
    Autoregressive Decoder Transformer generating 2D tile tokens
    conditioned on a spatial condition vector.
    """

    def __init__(
        self,
        condition_dim: int = 27,
        vocab_size: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.condition_proj = nn.Linear(condition_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding2D(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_custom_mask(self, size: int) -> torch.Tensor:
        """Causal triangular attention mask allowing condition token visibility."""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask[:, 0] = True
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

    def forward(self, condition: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        seq_len = src.size(1)
        cond_emb = self.condition_proj(condition).unsqueeze(1)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoder(seq_len, src.device)

        combined_src = torch.cat([cond_emb, src_emb], dim=1)
        mask = self.generate_custom_mask(seq_len + 1).to(src.device)
        output = self.transformer(combined_src, mask=mask)
        return self.fc_out(output[:, 1:, :])
