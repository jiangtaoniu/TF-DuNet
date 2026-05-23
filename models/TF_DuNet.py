from layers.decomposition import SeriesDecomp

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbeddingWoPos
from layers.StandardNorm import Normalize
from layers.MSDB import MSDB
from layers.ASST import ASST


class DFT_series_decomp(nn.Module):
    """
    Series decomposition using Discrete Fourier Transform (DFT).

    This module separates a time series into seasonal and trend components by
    transforming it into the frequency domain, filtering out low-amplitude
    frequencies (keeping the top k), and transforming back to the time domain.
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # Transform the real-valued time series to the frequency domain using Real Fast Fourier Transform
        xf = torch.fft.rfft(x)
        # Calculate the magnitude (amplitude) of each frequency component
        freq = abs(xf)
        # Remove the zero frequency (DC component) to ignore the global mean/trend
        freq[0] = 0
        # Identify the top 'k' frequency amplitudes indicative of dominant seasonal patterns
        top_k_freq, _ = torch.topk(freq, self.top_k)
        # Filter out noise/low-impact frequencies below the top 'k' threshold by zeroing them
        xf[freq <= top_k_freq.min()] = 0
        # Reconstruct the seasonal component via Inverse Real FFT
        x_season = torch.fft.irfft(xf)
        # The residual represents the slow-moving trend component
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal patterns across scales."""

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
            ) for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        # Ensure there are at least two elements before accessing index 1
        if len(season_list) < 2:
            return [out_high.permute(0, 2, 1)]

        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend patterns across scales."""

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** i)),
            ) for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        if len(trend_list_reverse) < 2:
            return [out_low.permute(0, 2, 1)]

        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


# --- Fusion Modules for Internal and Final Fusion ---
class ConcatProjectionFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d_model * 2, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        fused_result = self.fusion_mlp(torch.cat([x1, x2], dim=-1))
        return self.norm(x1 + fused_result)


class IndependentGatingFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_x1_linear = nn.Linear(d_model, d_model)
        self.gate_x2_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_x1 = torch.sigmoid(self.gate_x1_linear(x1))
        gate_x2 = torch.sigmoid(self.gate_x2_linear(x2))
        fused_result = gate_x1 * x1 + gate_x2 * x2
        return self.norm(x1 + fused_result)


class FiLMFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma_generator = nn.Linear(d_model, d_model)
        self.beta_generator = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):  # x1 is modulated, x2 is the modulator
        gamma = self.gamma_generator(x2)
        beta = self.beta_generator(x2)
        fused_result = gamma * x1 + beta
        return self.norm(x1 + fused_result)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout,
                                                     batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU() if activation == "gelu" else nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):  # x1 is Query, x2 is Key/Value
        attn_output, _ = self.cross_attention(query=x1, key=x2, value=x2)
        x1 = self.norm1(x1 + self.dropout(attn_output))
        ffn_output = self.ffn(x1)
        return self.norm2(x1 + self.dropout(ffn_output))


class StaticWeightedFusion(nn.Module):
    def __init__(self, num_features, initial_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), initial_alpha))
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x1, x2):
        alpha = torch.sigmoid(self.alpha)
        fused_result = alpha * x1 + (1 - alpha) * x2
        return self.norm(x1 + fused_result)


class GatingFusion(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.gate_generator = nn.Sequential(nn.Linear(2 * feature_dim, feature_dim), nn.Dropout(dropout), nn.Sigmoid())
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x1, x2):
        gate = self.gate_generator(torch.cat([x1, x2], dim=-1))
        fused_result = gate * x1 + (1 - gate) * x2
        return self.norm(x1 + fused_result)


class LearnableWeightedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x1, x2):
        proportions = F.softmax(torch.cat([self.w1, self.w2], dim=0), dim=0)
        fused_result = proportions[0] * x1 + proportions[1] * x2
        return self.norm(x1 + fused_result)


class CustomRangedWeightedFusion(nn.Module):
    def __init__(self, feature_dim, min_val=0.3, max_val=0.5):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.min_val = min_val
        self.range_width = max_val - min_val
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x1, x2):
        base_proportion = torch.sigmoid(self.p)
        alpha1 = base_proportion * self.range_width + self.min_val
        alpha2 = 1.0 - alpha1
        fused_result = alpha1 * x1 + alpha2 * x2
        return self.norm(x1 + fused_result)


class DecompositionBranch(nn.Module):
    """
    A branch of the model that performs time-series decomposition,
    multi-scale mixing, and spatial dependency modeling with MSDB.

    It extracts multiple scales of historical data, splits them into trend and seasonal
    parts, refines the features through multi-scale mixing, models graph correlations
    at each level, and subsequently fuses the resulting multi-scale representations.
    """

    def __init__(self, configs):
        super(DecompositionBranch, self).__init__()
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = SeriesDecomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('Unknown decomposition method')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(nn.Linear(configs.d_model, configs.d_ff), nn.GELU(),
                                             nn.Linear(configs.d_ff, configs.d_model))

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(nn.Linear(configs.d_model, configs.d_ff), nn.GELU(),
                                             nn.Linear(configs.d_ff, configs.d_model))
        self.norm = nn.LayerNorm(configs.d_model)

        self.msdb_blocks = nn.ModuleList()
        # Note: 'msdb_internal_fusion_method' should be set in your config
        if configs.msdb_internal_fusion_method == 'static_weighted':
            self.alpha_vectors = nn.ParameterList()

        for i in range(configs.top_k):
            self.msdb_blocks.append(
                MSDB(configs.c_out, configs.d_model, configs.conv_channel, configs.skip_channel,
                     configs.gcn_depth, configs.dropout, configs.propalpha, configs.node_dim))
            if configs.msdb_internal_fusion_method == 'static_weighted':
                self.alpha_vectors.append(nn.Parameter(torch.randn(configs.d_model)))

        self.msdb_internal_fusion_method = configs.msdb_internal_fusion_method
        self.msdb_internal_fusion_module = None
        if self.msdb_internal_fusion_method == 'concat':
            self.msdb_internal_fusion_module = ConcatProjectionFusion(configs.d_model, configs.dropout)
        elif self.msdb_internal_fusion_method == 'gate':
            self.msdb_internal_fusion_module = IndependentGatingFusion(configs.d_model)
        elif self.msdb_internal_fusion_method == 'film':
            self.msdb_internal_fusion_module = FiLMFusion(configs.d_model)
        elif self.msdb_internal_fusion_method == 'cross_attention':
            self.msdb_internal_fusion_module = CrossAttentionFusion(configs.d_model, configs.n_heads, configs.d_ff,
                                                                    configs.dropout, configs.activation)
        elif self.msdb_internal_fusion_method not in ['add', 'static_weighted']:
            raise ValueError(f"Unknown MSDB internal fusion method: {self.msdb_internal_fusion_method}")

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list, msdb_outputs = [], [], []

        for index, x in enumerate(x_list):
            msdb_out = self.msdb_blocks[index](x)
            msdb_outputs.append(msdb_out)

            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for idx, (ori, out_season, out_trend, msdb_out, length) in enumerate(zip(
                x_list, out_season_list, out_trend_list, msdb_outputs, length_list
        )):
            temporal_out = out_season + out_trend

            if self.msdb_internal_fusion_method == 'add':
                fused = self.norm(temporal_out + msdb_out)
            elif self.msdb_internal_fusion_method == 'static_weighted':
                raw_alpha = self.alpha_vectors[idx]
                channel_weights = torch.sigmoid(raw_alpha).view(1, 1, -1)
                fused = self.norm(channel_weights * temporal_out + (1 - channel_weights) * msdb_out)
            else:
                fused = self.msdb_internal_fusion_module(temporal_out, msdb_out)

            if self.channel_independence:
                fused = ori + self.out_cross_layer(fused)

            out_list.append(fused[:, :length, :])
        return out_list


class Model(nn.Module):
    """
    This class implements the Time-Frequency Dual Network (TF-DuNet) architecture.
    It is named 'Model' to be compatible with the experiment runner framework.

    The model employs a dual-branch architecture:
    1.  **ASST Branch**: An Adaptive Sequential Spatio-Temporal (ASST) block that uses
        self-attention to capture global spatio-temporal dependencies.
    2.  **Decomposition Branch**: A multi-layered branch that decomposes the time series
        into trend and seasonal components, processes them across multiple scales, and
        models spatial relationships using a dynamic graph-based MSDB module.

    The outputs of both branches are fused to produce the final forecast.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence

        # Branch 1: ASST (formerly STAEformer)
        # Note: Arguments passed here assume they are prefixed with 'asst_' in the config
        self.asst_branch = ASST(
            num_nodes=configs.c_out,
            in_steps=configs.seq_len,
            out_steps=configs.pred_len,
            output_dim=getattr(configs, 'asst_output_dim', 1),
            steps_per_day=getattr(configs, 'asst_steps_per_day', 288),
            input_dim=getattr(configs, 'asst_input_dim', 3),
            input_embedding_dim=getattr(configs, 'asst_input_embedding_dim', 24),
            tod_embedding_dim=getattr(configs, 'asst_tod_embedding_dim', 24),
            dow_embedding_dim=getattr(configs, 'asst_dow_embedding_dim', 24),
            spatial_embedding_dim=getattr(configs, 'asst_spatial_embedding_dim', 0),
            adaptive_embedding_dim=getattr(configs, 'asst_adaptive_embedding_dim', 80),
            feed_forward_dim=getattr(configs, 'asst_feed_forward_dim', 256),
            num_heads=getattr(configs, 'asst_n_heads', 4),
            num_layers=getattr(configs, 'asst_e_layers', 3),
            dropout=configs.dropout,
            use_mixed_proj=getattr(configs, 'asst_use_mixed_proj', True)
        )

        # Branch 2: Decomposition and MSDB
        self.decomposition_branch_layers = nn.ModuleList(
            [DecompositionBranch(configs) for _ in range(configs.e_layers)])

        self.preprocess = SeriesDecomp(configs.moving_avg)
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        embedding_channels = 1 if self.channel_independence == 1 else configs.enc_in
        self.enc_embedding = DataEmbeddingWoPos(embedding_channels, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=(configs.use_norm == 0))
            for _ in range(configs.down_sampling_layers + 1)
        ])

        # Final fusion logic between the two branches
        # Note: 'final_fusion_method' should be set in your config
        self.final_fusion_method = configs.final_fusion_method
        fusion_feature_dim = configs.c_out
        if self.final_fusion_method == 'add':
            self.final_fusion_module = None
        elif self.final_fusion_method == 'static_weighted':
            self.final_fusion_module = StaticWeightedFusion(fusion_feature_dim)
        elif self.final_fusion_method == 'concat':
            self.final_fusion_module = ConcatProjectionFusion(fusion_feature_dim, configs.dropout)
        elif self.final_fusion_method == 'gate':
            self.final_fusion_module = GatingFusion(fusion_feature_dim, configs.dropout)
        elif self.final_fusion_method == 'film':
            self.final_fusion_module = FiLMFusion(fusion_feature_dim)
        elif self.final_fusion_method == 'learnable_weighted':
            self.final_fusion_module = LearnableWeightedFusion(fusion_feature_dim)
        elif self.final_fusion_method == 'custom_ranged':
            self.final_fusion_module = CustomRangedWeightedFusion(fusion_feature_dim)
        elif self.final_fusion_method == 'cross_attention':
            num_heads = configs.n_heads if configs.c_out % configs.n_heads == 0 else 1
            self.final_fusion_module = CrossAttentionFusion(fusion_feature_dim, num_heads, dropout=configs.dropout)
        else:
            raise ValueError(f"Unknown final fusion method: {self.final_fusion_method}")

        self._build_task_specific_layers(configs)

    def _build_task_specific_layers(self, configs):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.predict_layers = nn.ModuleList([
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i), configs.pred_len)
                for i in range(configs.down_sampling_layers + 1)
            ])
            proj_out_features = 1 if self.channel_independence == 1 else configs.c_out
            self.projection_layer = nn.Linear(configs.d_model, proj_out_features, bias=True)

            if self.channel_independence == 0:
                self.out_res_layers = nn.ModuleList([
                    nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                              configs.seq_len // (configs.down_sampling_window ** i))
                    for i in range(configs.down_sampling_layers + 1)
                ])
                self.regression_layers = nn.ModuleList([
                    nn.Linear(configs.seq_len // (configs.down_sampling_window ** i), configs.pred_len)
                    for i in range(configs.down_sampling_layers + 1)
                ])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Main forecasting logic for the TF-DuNet model."""
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                # This block seems to have a shape mismatch. Assuming x_enc is 4D.
                B, T, N, C = x_enc.shape
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
            self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        # --- Branch 1: ASST ---
        asst_output = self.asst_branch(x_enc).squeeze(-1)

        # --- Branch 2: Decomposition ---
        x_enc_main = x_enc[:, :, :, 0]
        x_enc_scales, x_mark_scales = self.__multi_scale_process_inputs(x_enc_main, x_mark_enc)

        x_list, x_mark_list = [], []
        if x_mark_scales is not None:
            for i, (x, x_mark) in enumerate(zip(x_enc_scales, x_mark_scales)):
                B_scale, T_scale, N_scale = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B_scale * N_scale, T_scale, 1)
                    x_mark = x_mark.repeat(N_scale, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in enumerate(x_enc_scales):
                B_scale, T_scale, N_scale = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B_scale * N_scale, T_scale, 1)
                x_list.append(x)

        season_list, trend_list = self.pre_enc(x_list)

        enc_out_list = []
        if x_mark_scales is not None:
            for x, x_mark in zip(season_list, x_mark_list):
                enc_out_list.append(self.enc_embedding(x, x_mark))
        else:
            for x in season_list:
                enc_out_list.append(self.enc_embedding(x, None))

        for layer in self.decomposition_branch_layers:
            enc_out_list = layer(enc_out_list)

        # ================================== THE FIX IS HERE ==================================
        B, _, N, _ = x_enc.shape  # Unpack the original 4D tensor shape correctly
        # =====================================================================================

        decomp_decoded_list = self.decode_decomposition_branch(B, enc_out_list, season_list, trend_list)
        decomp_output = torch.stack(decomp_decoded_list, dim=-1).sum(-1)

        # --- Final Fusion ---
        if self.final_fusion_method == 'add':
            final_output = decomp_output + asst_output
        else:
            final_output = self.final_fusion_module(decomp_output, asst_output)

        return self.normalize_layers[0](final_output, 'denorm')

    def decode_decomposition_branch(self, B, enc_out_list, season_list, trend_list):
        dec_out_list = []
        if self.channel_independence == 1:
            for i, enc_out in enumerate(enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, trend_list)):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return x_list, None
        else:
            season_list, trend_list = [], []
            for x in x_list:
                season, trend = self.preprocess(x)
                season_list.append(season)
                trend_list.append(trend)
            return season_list, trend_list

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular', bias=False)
        else:
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else None

        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]
        x_mark_sampling_list = [x_mark_enc]

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            if x_mark_enc_mark_ori is not None:
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

        x_mark_scales = x_mark_sampling_list if x_mark_enc is not None else None
        return x_enc_sampling_list, x_mark_scales

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            # Other tasks like imputation, etc., would need to be refactored as well.
            # This simplified version focuses on the main forecasting task.
            raise ValueError(f"Task '{self.task_name}' is not yet implemented in this refactored version.")