import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.RADAR import RADAR
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Construct after the backbone so its initialization stays identical.
        self.radar = RADAR(configs.seq_len) if getattr(configs, 'use_radar', False) else None
        self.radar_target_only = getattr(configs, 'features', 'M') == 'MS'

    def _forecast_backbone(self, x_enc, x_mark_enc, x_dec, x_mark_dec, detach_mean=True):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True)
            if detach_mean:
                means = means.detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # Input saliency and imputer training need the pre-division tensor
            # saved by variance backward; an in-place division invalidates it.
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def _repair_input(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_radar_loss):
        if self.radar is None:
            return x_enc, x_enc.new_zeros(())

        def predictor(probe_x):
            # Clone covariates inside the grad-enabled probe, including when the
            # caller created them as inference tensors. Future values are unused.
            probe_marks = x_mark_enc.detach().clone() if x_mark_enc is not None else None
            forecast = self._forecast_backbone(probe_x, probe_marks, None, None, detach_mean=False)[0]
            return forecast[:, :, -1:] if self.radar_target_only else forecast

        return self.radar(x_enc, predictor, self, compute_reconstruction_loss=return_radar_loss)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        repaired, _ = self._repair_input(x_enc, x_mark_enc, x_dec, x_mark_dec, False)
        return self._forecast_backbone(repaired, x_mark_enc, x_dec, x_mark_dec)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_radar_loss=False):
        repaired, radar_loss = self._repair_input(x_enc, x_mark_enc, x_dec, x_mark_dec, return_radar_loss)
        dec_out, attns = self._forecast_backbone(repaired, x_mark_enc, x_dec, x_mark_dec)
        if return_radar_loss:
            # DataParallel gathers the per-replica auxiliary loss with forecasts.
            return dec_out[:, -self.pred_len:, :], radar_loss.unsqueeze(0)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
