import torch
import torch.nn as nn
from Models.interpretable_diffusion.Embed import DataEmbedding_wo_pos
from Models.interpretable_diffusion.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Models.interpretable_diffusion.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, 
        n_feat,     
        seq_len = 12,
        feat_len = 12,
        moving_avg = 25,
        dropout = 0.05,
        factor = 3,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        activation='GELU',
        output_attention = False,
        freq = 'h' ,
        embed_type = "timeF",
        **kwargs):
        super(Model, self).__init__()
        self.feat_len = feat_len
        self.label_len = feat_len//2
        self.pred_len = seq_len
        # self.output_attention = configs.output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)
        self.output_attention = output_attention
        d_ff = n_embd*4
        
        # [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(n_feat, n_embd, embed_type, freq,
                                                  dropout)
        self.dec_embedding = DataEmbedding_wo_pos(n_feat, n_embd, embed_type, freq,
                                                  dropout)
        
        if n_feat < 32 and seq_len < 64:
            kernel_size, padding = 1, 0
        else:
            kernel_size, padding = 5, 2
        

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        n_embd, n_heads),
                    n_embd,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layer_enc)
            ],
            norm_layer=my_Layernorm(n_embd)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        n_embd, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        n_embd, n_heads),
                    n_embd,
                    n_feat, 
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    # n_channel = self.label_len + self.pred_len
                )
                for l in range(n_layer_dec)
            ],
            norm_layer=my_Layernorm(n_embd),
            projection=nn.Linear(n_embd, n_feat, bias=True),
            # d_model = n_embd,
            # n_feat = n_feat
            
        )
    
        






    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,t,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec[:,-self.pred_len:,:]], dim=1)
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec[:,-self.pred_len:,:]], dim=1)
        
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        enc_out, attns = self.encoder(enc_out, t=t,attn_mask=enc_self_mask)
        # dec
        dec_in = torch.cat([seasonal_init[:, -self.label_len:, :], x_dec[:,-self.pred_len:,:]], dim=1)
        
        dec_in = self.dec_embedding(dec_in, x_mark_dec)
        
        seasonal_part, trend_part = self.decoder(dec_in, enc_out,t, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        
        dec_out = trend_part + seasonal_part
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out  # [B, L, D]
        
    

