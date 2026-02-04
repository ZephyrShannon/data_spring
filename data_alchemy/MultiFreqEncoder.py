import torch
from scipy.special import F
from torch import nn

from data_alchemy.ParallelStackedGRU import ParallelStackedGRU


class MultiFreqEncoderWithFiLM(nn.Module):
    """
    三频率层次化FiLM融合
    思路：低频→中频→高频，逐层调制
    """

    def __init__(
            self,
            # 各频率输入维度
            high_dim: int,  # 高频特征维度
            mid_dim: int,  # 中频特征维度
            low_dim: int,  # 低频特征维度

            # 各编码器配置
            hidden_dim: int = 256,
            num_layers: int = 2,
            parallelism: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()

        # ===== 低频编码器 =====
        self.low_encoder = ParallelStackedGRU(
            input_size=low_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism,
            dropout=dropout
        )

        # ===== 中频编码器（受低频调制） =====
        # 低频→中频的FiLM参数
        self.mid_gamma_from_low = nn.Linear(hidden_dim, mid_dim)
        self.mid_beta_from_low = nn.Linear(hidden_dim, mid_dim)

        self.mid_encoder = ParallelStackedGRU(
            input_size=mid_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism,
            dropout=dropout
        )

        # ===== 高频编码器（受中频调制） =====
        # 中频→高频的FiLM参数
        self.high_gamma_from_mid = nn.Linear(hidden_dim, high_dim)
        self.high_beta_from_mid = nn.Linear(hidden_dim, high_dim)

        self.high_encoder = ParallelStackedGRU(
            input_size=high_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism,
            dropout=dropout
        )

        # ===== 可选：直接低频→高频调制 =====
        self.high_gamma_from_low = nn.Linear(hidden_dim, high_dim)
        self.high_beta_from_low = nn.Linear(hidden_dim, high_dim)

        # ===== 最终融合层 =====
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(
            self,
            x_high: torch.Tensor,  # (B, L_high, D_high)
            x_mid: torch.Tensor,  # (B, L_mid, D_mid)
            x_low: torch.Tensor  # (B, L_low, D_low)
    ) -> torch.Tensor:
        # 1. 编码低频数据
        low_encoded, _ = self.low_encoder(x_low)  # (B, L_low, H)
        low_context = low_encoded[:, -1, :]  # (B, H) 取最后时间步

        # 2. 低频调制中频
        gamma_mid = self.mid_gamma_from_low(low_context).unsqueeze(1)  # (B, 1, D_mid)
        beta_mid = self.mid_beta_from_low(low_context).unsqueeze(1)  # (B, 1, D_mid)
        x_mid_modulated = gamma_mid * x_mid + beta_mid

        # 3. 编码调制后的中频
        mid_encoded, _ = self.mid_encoder(x_mid_modulated)  # (B, L_mid, H)
        mid_context = mid_encoded[:, -1, :]  # (B, H)

        # 4. 中频调制高频（主要路径）
        gamma_high_mid = self.high_gamma_from_mid(mid_context).unsqueeze(1)  # (B, 1, D_high)
        beta_high_mid = self.high_beta_from_mid(mid_context).unsqueeze(1)  # (B, 1, D_high)

        # 5. 可选：低频直接调制高频
        gamma_high_low = self.high_gamma_from_low(low_context).unsqueeze(1)  # (B, 1, D_high)
        beta_high_low = self.high_beta_from_low(low_context).unsqueeze(1)  # (B, 1, D_high)

        # 6. 双重调制：中频+低频共同调制高频
        x_high_modulated = (
                gamma_high_mid * gamma_high_low * x_high +  # 缩放相乘
                beta_high_mid + beta_high_low  # 偏移相加
        )

        # 7. 编码调制后的高频
        high_encoded, _ = self.high_encoder(x_high_modulated)  # (B, L_high, H)

        # 8. 三频率特征融合
        # 对齐序列长度（可选：使用池化或插值）
        low_pooled = F.adaptive_avg_pool1d(low_encoded.transpose(1, 2), x_high.shape[1]).transpose(1, 2)
        mid_pooled = F.adaptive_avg_pool1d(mid_encoded.transpose(1, 2), x_high.shape[1]).transpose(1, 2)

        # 拼接所有频率的特征
        fused = torch.cat([high_encoded, mid_pooled, low_pooled], dim=-1)  # (B, L_high, 3H)
        fused = self.fusion(fused)  # (B, L_high, H)

        return fused


class CrossFreqFiLMFusion(nn.Module):
    """
    双向交叉调制FiLM融合
    所有频率互相影响，形成完全连接的调制网络
    """

    def __init__(
            self,
            high_dim: int,
            mid_dim: int,
            low_dim: int,
            hidden_dim: int = 256,
            cross_attention_heads: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()

        # 各频率的基础编码器
        self.high_encoder = nn.GRU(high_dim, hidden_dim, batch_first=True)
        self.mid_encoder = nn.GRU(mid_dim, hidden_dim, batch_first=True)
        self.low_encoder = nn.GRU(low_dim, hidden_dim, batch_first=True)

        # 交叉注意力层（生成调制参数）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # FiLM参数生成网络
        self.film_param_generator = nn.ModuleDict({
            # 低频调制中频
            'low_to_mid': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, mid_dim * 2)  # gamma + beta
            ),
            # 低频调制高频
            'low_to_high': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, high_dim * 2)
            ),
            # 中频调制低频
            'mid_to_low': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, low_dim * 2)
            ),
            # 中频调制高频
            'mid_to_high': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, high_dim * 2)
            ),
            # 高频调制低频
            'high_to_low': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, low_dim * 2)
            ),
            # 高频调制中频
            'high_to_mid': nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, mid_dim * 2)
            ),
        })

        # 最终融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_high, x_mid, x_low):
        B, L_high, _ = x_high.shape
        _, L_mid, _ = x_mid.shape
        _, L_low, _ = x_low.shape

        # 1. 初始编码（无调制）
        h_high, _ = self.high_encoder(x_high)  # (B, L_high, H)
        h_mid, _ = self.mid_encoder(x_mid)  # (B, L_mid, H)
        h_low, _ = self.low_encoder(x_low)  # (B, L_low, H)

        # 2. 提取上下文特征（最后一个时间步）
        ctx_high = h_high[:, -1, :]  # (B, H)
        ctx_mid = h_mid[:, -1, :]  # (B, H)
        ctx_low = h_low[:, -1, :]  # (B, H)

        # 3. 交叉注意力生成融合上下文
        # 将所有上下文拼接作为查询
        all_context = torch.stack([ctx_low, ctx_mid, ctx_high], dim=1)  # (B, 3, H)

        # 自注意力融合
        fused_context, _ = self.cross_attention(
            all_context, all_context, all_context
        )  # (B, 3, H)

        ctx_low_fused = fused_context[:, 0, :]  # 融合后的低频上下文
        ctx_mid_fused = fused_context[:, 1, :]  # 融合后的中频上下文
        ctx_high_fused = fused_context[:, 2, :]  # 融合后的高频上下文

        # 4. 生成所有调制参数
        film_params = {}

        # 低频→中频
        low_mid_input = torch.cat([ctx_low_fused, ctx_mid_fused], dim=-1)
        low_mid_params = self.film_param_generator['low_to_mid'](low_mid_input)
        film_params['low_to_mid_gamma'] = low_mid_params[:, :x_mid.shape[-1]]
        film_params['low_to_mid_beta'] = low_mid_params[:, x_mid.shape[-1]:]

        # 类似地生成其他方向的调制参数...

        # 5. 应用所有调制（可以迭代多次）
        # 第一轮调制
        x_mid_mod1 = (film_params['low_to_mid_gamma'].unsqueeze(1) * x_mid +
                      film_params['low_to_mid_beta'].unsqueeze(1))

        # 重新编码调制后的特征
        h_mid_mod1, _ = self.mid_encoder(x_mid_mod1)
        ctx_mid_mod1 = h_mid_mod1[:, -1, :]

        # 第二轮调制（中频→高频，使用更新后的中频上下文）
        mid_high_input = torch.cat([ctx_mid_mod1, ctx_high_fused], dim=-1)
        mid_high_params = self.film_param_generator['mid_to_high'](mid_high_input)

        # 6. 最终调制和融合
        x_high_final = (mid_high_params[:, :x_high.shape[-1]].unsqueeze(1) * x_high +
                        mid_high_params[:, x_high.shape[-1]:].unsqueeze(1))

        # 最终编码
        h_high_final, _ = self.high_encoder(x_high_final)
        h_mid_final, _ = self.mid_encoder(x_mid_mod1)
        h_low_final, _ = self.low_encoder(x_low)

        fused = torch.cat([h_high_final, h_mid_aligned, h_low_aligned], dim=-1)
        fused = self.fusion_layer(fused)

        return fused


class HierarchicalFiLMFusion(nn.Module):
    """
    实用版：层级调制 + 门控融合
    更适合实际部署
    """

    def __init__(
            self,
            # 输入维度
            high_dim: int,
            mid_dim: int,
            low_dim: int,

            # 编码器配置
            encoder_hidden: int = 128,

            # FiLM配置
            film_hidden: int = 64,

            # 输出配置
            output_dim: int = 1
    ):
        super().__init__()

        # ========== 层级1：低频特征提取 ==========
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden)
        )

        # ========== 层级2：低频调制中频 ==========
        self.low_to_mid_film = nn.Sequential(
            nn.Linear(encoder_hidden, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, mid_dim * 2)  # gamma + beta
        )

        self.mid_encoder = nn.Sequential(
            nn.Linear(mid_dim, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden)
        )

        # ========== 层级3：中低频共同调制高频 ==========
        self.context_fusion = nn.Sequential(
            nn.Linear(encoder_hidden * 2, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, film_hidden)
        )

        self.context_to_high_film = nn.Linear(film_hidden, high_dim * 2)

        self.high_encoder = nn.GRU(
            high_dim, encoder_hidden,
            batch_first=True,
            bidirectional=False
        )

        # ========== 门控融合层 ==========
        self.gate_network = nn.Sequential(
            nn.Linear(encoder_hidden * 3, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, 3),  # 3个门的权重
            nn.Softmax(dim=-1)
        )

        # ========== 最终输出层 ==========
        self.output_layer = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden // 2),
            nn.ReLU(),
            nn.Linear(encoder_hidden // 2, output_dim)
        )

    def forward(self, x_high, x_mid, x_low):
        """
        x_high: (B, L_high, D_high) - 高频序列
        x_mid: (B, D_mid) - 中频特征（已聚合）
        x_low: (B, D_low) - 低频特征（已聚合）
        """
        # 1. 处理低频特征
        h_low = self.low_encoder(x_low)  # (B, H)

        # 2. 低频调制中频
        film_params_mid = self.low_to_mid_film(h_low)
        gamma_mid = film_params_mid[:, :x_mid.shape[-1]]  # (B, D_mid)
        beta_mid = film_params_mid[:, x_mid.shape[-1]:]  # (B, D_mid)

        # 调制中频特征
        x_mid_mod = gamma_mid * x_mid + beta_mid
        h_mid = self.mid_encoder(x_mid_mod)  # (B, H)

        # 3. 融合中低频上下文
        context = torch.cat([h_low, h_mid], dim=-1)  # (B, 2H)
        context_fused = self.context_fusion(context)  # (B, F)

        # 4. 上下文调制高频
        film_params_high = self.context_to_high_film(context_fused)
        gamma_high = film_params_high[:, :x_high.shape[-1]]  # (B, D_high)
        beta_high = film_params_high[:, x_high.shape[-1]:]  # (B, D_high)

        # 调制高频序列
        gamma_high = gamma_high.unsqueeze(1)  # (B, 1, D_high)
        beta_high = beta_high.unsqueeze(1)  # (B, 1, D_high)
        x_high_mod = gamma_high * x_high + beta_high

        # 编码调制后的高频
        h_high_seq, _ = self.high_encoder(x_high_mod)  # (B, L_high, H)
        h_high = h_high_seq[:, -1, :]  # 取最后一个时间步 (B, H)

        # 5. 门控融合
        all_features = torch.cat([h_low, h_mid, h_high], dim=-1)  # (B, 3H)
        gate_weights = self.gate_network(all_features)  # (B, 3)

        # 加权融合
        features_stacked = torch.stack([h_low, h_mid, h_high], dim=1)  # (B, 3, H)
        weighted_features = (features_stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)

        # 6. 最终输出
        output = self.output_layer(weighted_features)

        return output, gate_weights  # 返回门控权重用于解释
