import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, seq_len, d_model).
        """
        batch = Q.shape[0]
        Q = (Q @ self.W_q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = (K @ self.W_k).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (V @ self.W_v).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 修改 1：加上了逗号 -2, -1
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        
        # 修改 2：将小写 v 改为大写 V
        attn = weights @ V 
        
        concat = attn.transpose(2, 1).contiguous().view(batch, -1, self.num_heads * self.d_k)
        return concat @ self.W_o