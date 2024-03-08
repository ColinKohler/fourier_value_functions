import torch
import torch.nn as nn
import torch.nn.functional as F

import escnn
from escnn import gspaces
from escnn import nn as enn
from escnn import group

import math

from fail.model.layers import SO2MLP


def scaledDotProduct(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create matrix of [S, D] representing the positional embeding for max len inputs
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SO2MultiheadAttention(nn.Module):
    def __init__(self, in_type, L, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.in_type = in_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.L = L
        G = group.so2_group()
        self.irreps_dim = G.bl_regular_representation(L=self.L).size

        self.qkv_proj = SO2MLP(
            in_type, in_type, [3 * embed_dim], [self.L], act_out=False
        )
        self.o_proj = SO2MLP(in_type, in_type, [embed_dim], [self.L], act_out=False)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, embed_dim = x.size()  # [B, S, D]
        x = self.in_type(x.view(batch_size * seq_len, embed_dim))
        qkv = self.qkv_proj(x).tensor

        # Seperate Q, K, V
        qkv = qkv.reshape(
            batch_size, seq_len, self.num_heads, 3 * self.irreps_dim * self.head_dim
        )
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, S, L*D]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaledDotProduct(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [B, S, H, L*D]
        values = values.reshape(batch_size * seq_len, embed_dim)
        o = self.o_proj(self.in_type(values))
        # o = o.tensor.reshape(batch_size, seq_len, -1)

        if return_attention:
            return o, attention
        else:
            return o


class SO2EncoderBlock(nn.Module):
    def __init__(self, in_type, L, in_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.in_type = in_type
        self.L = L

        self.attn = SO2MultiheadAttention(in_type, L, in_dim, num_heads)
        self.mlp = SO2MLP(
            in_type,
            in_type,
            [hidden_dim, in_dim],
            [self.L, self.L],
            dropout=dropout,
            act_out=False,
        )

        # self.norm1 = nn.LayerNorm(in_dim)
        # self.norm2 = nn.LayerNorm(in_dim)
        self.dropout1 = enn.FieldDropout(self.in_type)
        self.dropout2 = enn.FieldDropout(self.mlp.out_type, dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()  # [B, S, D]

        attn_out = self.attn(x, mask=mask)
        x = self.in_type(x.view(batch_size * seq_len, -1)) + self.dropout1(attn_out)
        # x = self.norm1(x)
        x = x + self.dropout2(self.mlp(x))
        # x = self.norm2(x)

        return x


class SO2TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.in_type = block_args["in_type"]

        self.layers = nn.ModuleList(
            [SO2EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()  # [B, S, D]

        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if i < len(self.layers) - 1:
                x = x.tensor.view(batch_size, seq_len, -1)
        return x

    def getAttnMaps(self, x, mask=None):
        attn_maps = list()
        for layer in self.layers:
            _, attn_map = layer.attn(x, mask=mask, return_attention=True)
            attn_maps.append(attn_map)
            x = layer(x)
        return attn_maps


class SO2Transformer(nn.Module):
    def __init__(
        self,
        in_type,
        L,
        model_dim,
        out_dim,
        num_heads,
        num_layers,
        dropout=0.0,
        input_dropout=0.0,
    ):
        super().__init__()

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.L = L
        self.in_type = in_type

        t = self.G.bl_regular_representation(L=self.L)
        model_type = enn.FieldType(self.gspace, [t] * model_dim)
        self.out_type = model_type
        self.input_net = enn.SequentialModule(
            enn.FieldDropout(in_type, input_dropout), enn.Linear(in_type, model_type)
        )
        # self.pos_enc = PositionalEncoding(d_model=model_dim)
        self.transformer = SO2TransformerEncoder(
            num_layers=num_layers,
            in_type=model_type,
            L=L,
            in_dim=model_dim,
            hidden_dim=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.out = SO2MLP(
            model_type,
            self.out_type,
            [model_dim, out_dim],
            [self.L, self.L],
            act_out=False,
            dropout=dropout,
        )

    def forward(self, x, mask=None, add_pos_enc=True):
        batch_size, seq_len, in_dim = x.size()  # [B, S, D]

        x = self.in_type(x.view(batch_size * seq_len, -1))
        x = self.input_net(x).tensor.view(batch_size, seq_len, -1)
        # if add_pos_enc:
        #    x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        x = self.out(x)
        return x

    def getAttnMaps(self, x, mask=None, add_pos_enc=True):
        x = self.input_net(x)
        # if add_pos_enc:
        #    x = self.pos_enc(x)
        attn_maps = self.transformer.getAttnMaps(x, mask=mask)
        return attn_maps
