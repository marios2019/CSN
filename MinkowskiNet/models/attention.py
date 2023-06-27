import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME


class MultiHeadAttention(Module):
    """
      Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.norm(q)

        return q, attn


class ScaledDotProductAttention(Module):
    """
      Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ScaledDotProduct(Module):
    """
      Scaled dot-product
    """

    def __init__(self, temperature):
        super(ScaledDotProduct, self).__init__()

        self.temperature = temperature

    def forward(self, q, k):
        sparse = False
        if isinstance(q, ME.SparseTensor):
            assert (isinstance(k, ME.SparseTensor))
            output = torch.matmul(q.F, k.F.T)
            sparse = True
        else:
            if q.ndim == 2:
                q = q.unsqueeze(0)
            if k.ndim == 2:
                k = k.unsqueeze(0)
            output = torch.bmm(q, k.permute(0, 2, 1))

        output /= self.temperature

        if sparse:
            return ME.SparseTensor(
                output,
                coordinate_map_key=q.coordinate_map_key,
                coordinate_manager=q.coordinate_manager)
        return output

    def __repr__(self):
        s = '(temperature={})'.format(self.temperature)

        return self.__class__.__name__ + s
