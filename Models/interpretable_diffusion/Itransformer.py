import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Models.interpretable_diffusion.Embed import DataEmbedding_inverted
from Models.interpretable_diffusion.model_utils import AdaLayerNorm
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(
                mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.ln1 = AdaLayerNorm(d_model)

    def forward(self, x, t, attn_mask=None, tau=None, delta=None, label_emb=None):
        # print(x.size(),t.size())
        x = self.ln1(x, t, label_emb)
        # print(x.size())
        # exit()
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class DecoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.ln1 = AdaLayerNorm(d_model)

    def forward(self, x, t, enc, attn_mask=None, tau=None, delta=None, label_emb=None):
        x = self.ln1(x, t, label_emb)
        new_x, attn = self.attention(
            x, enc, enc,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, t, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, t, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, t, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, t, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Decoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Decoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, t, enc, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(
                    x, t, enc, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, t, enc, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x, t, enc, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ITransformer(nn.Module):
    def __init__(
        self,
        n_feat,
        seq_len,
        feat_len,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        embed_type="timeF",
        max_len=2048,
        freq='h',
        dropout=0.05,
        factor=3,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.seq_len = feat_len
        self.pred_len = seq_len
        self.output_attention = False
        self.use_norm = True
        d_ff = n_embd*4
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(feat_len+seq_len, n_embd, embed_type, freq,
                                                    dropout)
        # self.dec_embedding = DataEmbedding_inverted(feat_len//2+seq_len, n_embd, embed_type, freq,
        #                                            dropout)
        self.len = self.pred_len + self.seq_len//2

        n_layer_enc = 3
        n_layer_dec = 1
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=self.output_attention), n_embd, n_heads),
                    n_embd,
                    d_ff,
                    dropout=dropout,
                    activation=block_activate
                ) for l in range(n_layer_enc)
            ],
            norm_layer=torch.nn.LayerNorm(n_embd)
        )
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, factor, attention_dropout=dropout,
        #                               output_attention=self.output_attention), n_embd, n_heads),
        #             n_embd,
        #             d_ff,
        #             dropout=dropout,
        #             activation=block_activate
        #         ) for l in range(n_layer_dec)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(n_embd)
        # )
        self.projector = nn.Linear(n_embd, self.len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, t):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # x_enc[:,-self.seq_len//2:,-1] = x_dec[:,:self.seq_len//2,-1]
        # print(x_enc.size(),x_dec.size(),self.pred_len)
        # exit()

        input = torch.concat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        input_mark = torch.concat(
            [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Embedding

        # # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.enc_embedding(input, input_mark)
        # # torch.Size([48, 866, 512])
        # if x_mark_dec.size(1)==self.pred_len:

        #     x_mark_dec = torch.cat([x_mark_enc[:,self.seq_len//2:,:],x_mark_dec],dim=1)
        # dec = self.dec_embedding(x_dec, x_mark_dec)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        # enc_out, attns = self.encoder(enc_out, t,attn_mask=None)
        enc_out, attns = self.encoder(enc_out, t, attn_mask=None)
        # enc_out, attns = self.decoder(enc_out,t,enc,attn_mask=None)
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(
            0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.len, 1))
            dec_out = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, t, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, t)

        return dec_out  # [:, -self.pred_len:, :]  # [B, L, D]
