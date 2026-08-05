import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-LayerNorm (modern standard)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        mask: optional attention mask (usually not needed if using is_causal=True)
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # === Self-Attention (Pre-LN) ===
        x = self.ln1(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = rearrange(qkv, 'b t (three h d) -> three b h t d', three=3, h=self.n_heads, d=self.head_dim)

        # Scaled dot-product attention (modern & efficient)
        # is_causal=True creates the upper triangular mask automatically
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,           # Very important for decoder/LLM style
        )

        # Merge heads back
        attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        x = residual + attn_output

        # === Feed-Forward Network (Pre-LN) ===
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ====================== Usage Example ======================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 2
    seq_len = 16
    d_model = 128
    n_heads = 8
    d_ff = 512

    model = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer block works!")
