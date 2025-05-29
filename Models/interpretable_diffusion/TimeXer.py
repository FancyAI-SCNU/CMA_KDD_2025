import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from Models.interpretable_diffusion.model_utils import AdaLayerNorm
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        # print(nf)
        # exit()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask,
                      cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.ln1 = AdaLayerNorm(d_model)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))

        # cross = self.ln1(cross,t)

        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self,
                 n_feat,
                 seq_len,
                 feat_len,
                 n_layer_enc=5,
                 n_layer_dec=14,
                 n_embd=1024,
                 n_heads=8,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 block_activate='GELU',
                 embed_type="timeF",
                 task_name='long_term_forecast',
                 features='M',
                 d_ff=2048,
                 freq='h',
                 dropout=0.05,
                 factor=3,
                 conv_params=None,
                 **kwargs):
        super(Model, self).__init__()
        self.task_name = task_name
        self.features = features
        self.seq_len = feat_len
        self.pred_len = seq_len
        self.use_norm = True
        self.patch_len = 16
        # 60 30 90
        self.patch_num = int(self.seq_len // self.patch_len)  # +3# +3#+3#+2

        self.len = self.pred_len + self.seq_len//2
        self.n_vars = 1 if self.features == 'MS' else n_feat
        # Embedding

        self.en_embedding = EnEmbedding(
            self.n_vars, n_embd, self.patch_len, dropout)

        self.ex_embedding = DataEmbedding_inverted(self.seq_len, n_embd, embed_type, freq,
                                                   dropout)
        # self.ln1 = AdaLayerNorm(n_embd)
        # Encoder-only architecture

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        n_embd, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        n_embd, n_heads),
                    n_embd,
                    d_ff,
                    dropout=dropout,
                    activation=block_activate,
                )
                for l in range(n_layer_enc)
            ],
            norm_layer=torch.nn.LayerNorm(n_embd)
        )
        self.head_nf = n_embd * (self.patch_num + 1)
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
        #                         head_dropout=configs.dropout)

        self.head = FlattenHead(n_feat, self.head_nf, self.len,
                                head_dropout=dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # TODO:
        input = torch.concat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        input_mark = torch.concat(
            [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        en_embed, n_vars = self.en_embedding(
            input[:, :, -1].unsqueeze(-1).permute(0, 2, 1))

        ex_embed = self.ex_embedding(input[:, :, :-1], input_mark)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.len, 1))
            dec_out = dec_out + \
                (means[:, 0, -1:].unsqueeze(1).repeat(1, self.len, 1))

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # TODO:

        input = torch.concat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        input_mark = torch.concat(
            [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        input = x_enc
        input_mark = x_mark_enc
        en_embed, n_vars = self.en_embedding(input.permute(0, 2, 1))

        ex_embed = self.ex_embedding(input, input_mark)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.len, 1))
            dec_out = dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, self.len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(
                    x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out  # [:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(
                    x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out  # [:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None
