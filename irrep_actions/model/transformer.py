import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.resetParameters()

    def resetParameters(self):
        """Original Transformer uses xavier initialization & zero biases."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, embed_dim = x.size()  # [B, S, D]
        qkv = self.qkv_proj(x)

        # Seperate Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, S, D]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaledDotProduct(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [B, S, H. D]
        values = values.reshape(batch_size, seq_len, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()

        self.attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        x = x + self.dropout(self.mlp(x))
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def getAttnMaps(self, x, mask=None):
        attn_maps = list()
        for layer in self.layers:
            _, attn_map = layer.attn(x, mask=mask, return_attention=True)
            attn_maps.append(attn_map)
            x = layer(x)
        return attn_maps


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        out_dim,
        num_heads,
        num_layers,
        dropout=0.0,
        input_dropout=0.0,
    ):
        super().__init__()

        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout), nn.Linear(input_dim, model_dim)
        )
        self.pos_enc = PositionalEncoding(d_model=model_dim)
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            hidden_dim=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, out_dim),
        )

    def forward(self, x, mask=None, add_pos_enc=True):
        x = self.input_net(x)
        if add_pos_enc:
            x = self.pos_enc(x)
        x = self.transformer(x, mask=mask)
        x = self.out(x)
        return x

    def getAttnMaps(self, x, mask=None, add_pos_enc=True):
        x = self.input_net(x)
        if add_pos_enc:
            x = self.pos_enc(x)
        attn_maps = self.transformer.getAttnMaps(x, mask=mask)
        return attn_maps
