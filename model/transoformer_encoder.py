import torch
import torch.nn as nn

class ClauseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        key_padding_mask = ~mask
        h = self.norm1(x)
        h, attn_weights = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        h = self.drop1(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights
        
