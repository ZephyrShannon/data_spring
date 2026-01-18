import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


# ==============================
# 1. AdditiveAttention (保留，用于 pooling)
# ==============================
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_x = torch.tanh(self.proj(x))
        scores = torch.matmul(proj_x, self.v)
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        context = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return context, attn


# ==============================
# 2. ParallelStackedGRU (你的原始实现，复用)
# ==============================
class ParallelStackedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        parallelism_factor: int,
        dropout: float = 0.0
    ):
        super().__init__()
        if hidden_size % parallelism_factor != 0:
            raise ValueError("hidden_size must be divisible by parallelism_factor")
        self.hidden_per_slice = hidden_size // parallelism_factor
        self.gru_slices = nn.ModuleList([
            nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_per_slice,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(parallelism_factor)
        ])

    def forward(self, x: torch.Tensor):
        outputs = []
        for gru in self.gru_slices:
            out, _ = gru(x)
            outputs.append(out)
        return torch.cat(outputs, dim=-1), None


# ==============================
# 3. HighFreqEncoderWithFiLM (不变)
# ==============================
class HighFreqEncoderWithFiLM(nn.Module):
    def __init__(
        self,
        high_input_dim: int,
        mid_context_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        parallelism_factor: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gamma_proj = nn.Linear(mid_context_dim, high_input_dim)
        self.beta_proj = nn.Linear(mid_context_dim, high_input_dim)
        self.gru_encoder = ParallelStackedGRU(
            input_size=high_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor,
            dropout=dropout
        )

    def forward(self, x_high: torch.Tensor, mf_context: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(mf_context).unsqueeze(1)
        beta = self.beta_proj(mf_context).unsqueeze(1)
        x_modulated = gamma * x_high + beta
        hf_seq, _ = self.gru_encoder(x_modulated)
        return hf_seq


# ==============================
# 4. 单个 Mid-Freq Expert (现在用 ParallelStackedGRU + Pooling)
# ==============================
class MidFreqExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        parallelism_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gru = ParallelStackedGRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor,
            dropout=dropout
        )
        self.pooler = AdditiveAttention(hidden_dim)

    def forward(self, x_mid: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.gru(x_mid)          # (B, T, H)
        context, _ = self.pooler(seq_out)     # (B, H)
        return context


class LowFreqDualPathEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_experts: int = 4,
            num_layers: int = 2,
            parallelism_factor: int = 2,
            dropout: float = 0.1,
            use_regularization: bool = False,  # 是否使用正则化
            regularization_lambda: float = 0.01,  # 正则化强度
            death_threshold: float = 0.01  # 专家死亡阈值
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_regularization = use_regularization
        self.regularization_lambda = regularization_lambda
        self.death_threshold = death_threshold

        # GRU编码器
        self.gru = ParallelStackedGRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor,
            dropout=dropout
        )

        # 池化层
        self.pooler = AdditiveAttention(hidden_dim)

        # 改进的路由头：使用多种池化特征
        # 输入维度：平均池化 + 最大池化 + 末端状态 + 注意力上下文
        self.pooling_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

        # 跟踪专家使用统计（用于validation阶段）
        self.register_buffer('expert_usage_stats', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0))

        # 训练状态跟踪
        self.register_buffer('epoch_expert_usage', torch.zeros(num_experts))
        self.register_buffer('epoch_samples', torch.tensor(0))
        self.current_epoch = 0

    def _compute_router_features(self, seq_out: torch.Tensor) -> torch.Tensor:
        """
        计算更全面的路由特征
        seq_out: (B, T, H)
        return: (B, H*4)
        """
        batch_size, seq_len, hidden_dim = seq_out.shape

        # 1. 平均池化：捕捉整体趋势
        avg_pool = seq_out.mean(dim=1)  # (B, H)

        # 2. 最大池化：捕捉关键事件/显著特征
        max_pool, _ = seq_out.max(dim=1)  # (B, H)

        # 3. 末端状态：最新动态
        last_hidden = seq_out[:, -1, :]  # (B, H)

        # 4. 注意力池化：自适应加权
        attn_context, _ = self.pooler(seq_out)  # (B, H)

        # 拼接所有特征
        combined = torch.cat([avg_pool, max_pool, last_hidden, attn_context], dim=-1)

        # 投影到统一维度
        router_features = self.pooling_proj(combined)

        return router_features

    def forward(
            self,
            x_low: torch.Tensor,
            return_regularization: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_low: (B, T, D) 低频率输入
            return_regularization: 是否返回正则化损失

        Returns:
            expert_weights: (B, E) 专家权重
            reg_loss: 正则化损失（可选）
        """
        # 1. GRU编码
        seq_out, _ = self.gru(x_low)  # (B, T, H)

        # 2. 计算路由特征（使用多种池化组合）
        router_features = self._compute_router_features(seq_out)

        # 3. 路由决策
        router_logits = self.router_head(router_features)  # (B, E)

        # 使用softmax获取专家权重
        expert_weights = F.softmax(router_logits, dim=-1)
        entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(-1).mean()
        print(f"Routing Entropy: {entropy.item():.3f}")

        # 4. 更新使用统计（只在validation阶段或训练时跟踪）
        if not self.training or return_regularization:
            with torch.no_grad():
                # 计算每个专家的使用率（soft counting）
                batch_usage = expert_weights.mean(dim=0)  # (E,)
                batch_size = expert_weights.size(0)

                # 更新epoch统计
                self.epoch_expert_usage += batch_usage * batch_size
                self.epoch_samples += batch_size

                # 如果是validation模式，更新全局统计
                if not self.training:
                    self.expert_usage_stats += batch_usage * batch_size
                    self.total_samples += batch_size

        # 5. 计算正则化损失（如果启用）
        reg_loss = None
        if return_regularization and self.use_regularization:
            reg_loss = self._compute_regularization_loss(expert_weights)

        return expert_weights, reg_loss

    def _compute_regularization_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        计算正则化损失：
        1. 鼓励稀疏性（L2正则）
        2. 防止专家死亡
        """
        batch_size = expert_weights.size(0)

        # 计算当前批次的专家使用率
        expert_usage = expert_weights.mean(dim=0)  # (E,)

        # 1. L2正则：鼓励稀疏选择
        # 对每个样本的专家权重进行L2惩罚，鼓励少数专家被选中
        sparsity_loss = torch.mean(torch.sum(expert_weights ** 2, dim=1))  # (B,) -> scalar

        # 2. 死亡惩罚：防止专家完全死亡
        # 使用ReLU确保只有当使用率低于阈值时才惩罚
        death_penalty = F.relu(self.death_threshold - expert_usage)  # (E,)
        death_loss = death_penalty.sum()  # scalar

        # 组合损失
        reg_loss = self.regularization_lambda * (
                sparsity_loss + 0.5 * death_loss  # 给死亡惩罚较低权重
        )

        return reg_loss

    def get_expert_usage_stats(self, reset: bool = False) -> Dict[str, torch.Tensor]:
        """
        获取专家使用统计（用于validation阶段输出）

        Args:
            reset: 是否重置统计

        Returns:
            包含统计信息的字典
        """
        if self.total_samples.item() == 0:
            # 如果没有统计数据，使用epoch统计
            if self.epoch_samples.item() > 0:
                total_usage = self.epoch_expert_usage
                total_samples = self.epoch_samples
            else:
                return {
                    'expert_usage': torch.zeros(self.num_experts),
                    'usage_ratio': torch.zeros(self.num_experts),
                    'entropy': torch.tensor(0.0),
                    'min_usage': torch.tensor(0.0),
                    'max_usage': torch.tensor(0.0),
                    'is_balanced': torch.tensor(False)
                }
        else:
            total_usage = self.expert_usage_stats
            total_samples = self.total_samples

        # 计算平均使用率
        avg_usage = total_usage / total_samples  # (E,)

        # 计算使用比例（归一化到[0,1]）
        usage_ratio = avg_usage / avg_usage.sum()

        # 计算分布熵（衡量均衡程度）
        # 避免log(0)
        #epsilon = 1e-10
        safe_usage = usage_ratio + 1e-10
        entropy = -torch.sum(safe_usage * torch.log(safe_usage)) / torch.log(torch.tensor(self.num_experts))

        # 检查是否有专家死亡
        min_usage = avg_usage.min()
        max_usage = avg_usage.max()
        is_balanced = min_usage > self.death_threshold

        stats = {
            'expert_usage': avg_usage.clone(),
            'usage_ratio': usage_ratio.clone(),
            'entropy': entropy,
            'min_usage': min_usage,
            'max_usage': max_usage,
            'is_balanced': is_balanced,
            'total_samples': total_samples.clone()
        }

        # 如果需要重置统计
        if reset:
            self.expert_usage_stats.zero_()
            self.total_samples.zero_()

        return stats

    def update_epoch(self, epoch: int) -> None:
        """
        更新epoch，并根据统计决定是否启用正则化

        Args:
            epoch: 当前epoch
        """
        self.current_epoch = epoch

        # 检查上一个epoch的专家使用情况
        if self.epoch_samples.item() > 0:
            epoch_usage = self.epoch_expert_usage / self.epoch_samples

            # 检查是否有专家死亡（使用率<1%）
            min_usage = epoch_usage.min()
            has_dead_expert = min_usage < self.death_threshold

            # 如果有专家死亡，启用或增强正则化
            if has_dead_expert and not self.use_regularization:
                print(f"[Epoch {epoch}] 检测到专家死亡（最小使用率={min_usage:.3f}），启用正则化")
                self.use_regularization = True
                self.regularization_lambda = 0.01  # 温和的正则化
            elif has_dead_expert and self.regularization_lambda < 0.05:
                # 如果已有正则化但仍有死亡，增强正则化
                print(f"[Epoch {epoch}] 专家仍死亡，增强正则化强度")
                self.regularization_lambda = min(0.05, self.regularization_lambda * 1.5)

            # 重置epoch统计
            self.epoch_expert_usage.zero_()
            self.epoch_samples.zero_()

            # 输出统计信息
            self._print_epoch_stats(epoch, epoch_usage, has_dead_expert)

    def _print_epoch_stats(self, epoch: int, usage: torch.Tensor, has_dead_expert: bool) -> None:
        """打印epoch统计信息"""
        print(f"\n=== Epoch {epoch} 专家使用统计 ===")
        print(f"专家使用率: {usage.tolist()}")
        print(f"最小使用率: {usage.min().item():.4f}")
        print(f"最大使用率: {usage.max().item():.4f}")
        print(f"使用率标准差: {usage.std().item():.4f}")
        print(f"是否有专家死亡: {has_dead_expert}")
        print(f"正则化启用: {self.use_regularization}")
        if self.use_regularization:
            print(f"正则化强度: {self.regularization_lambda}")
        print("=" * 40)


class MultiScaleClassificationHeads(nn.Module):
    def __init__(
        self,
        mf_dim: int,
        hf_dim: int,
        class_config: Dict[str, Dict[str, int]],
        scales: List[str],
        hidden_dim: int = 128
    ):
        super().__init__()
        self.scales = scales
        self.class_config = class_config
        self.hf_pooler = AdditiveAttention(hf_dim)

        # 🔥 关键：fusion only from mf + hf
        self.fusion_dim = mf_dim + hf_dim

        self.heads = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1) # 只有一个
            ) for scale in self.scales
        })

    def forward(
        self,
        mf_context: torch.Tensor,
        hf_seq: torch.Tensor
    ) -> torch.Tensor:
        hf_context, _ = self.hf_pooler(hf_seq)
        fused = torch.cat([mf_context, hf_context], dim=-1)  # (B, mf+hf)
        outputs = [self.heads[scale](fused) for scale in self.scales]
        return torch.cat(outputs, dim=-1)


# 修改后的主模型，集成新的LowFreqDualPathEncoder
class ThreeLayerMoEWithSmartRouting(nn.Module):
    def __init__(
            self,
            low_input_dim: int,
            mid_input_dim: int,
            high_input_dim: int,
            class_config: Dict[str, Dict[str, int]],
            start_scale: int = 0,
            end_scale: int =0,
            low_hidden: int = 64,
            low_layers: int = 2,
            low_parallel: int = 2,
            mid_hidden: int = 128,
            mid_layers: int = 2,
            mid_parallel: int = 2,
            num_mid_experts: int = 4,
            high_hidden: int = 256,
            high_layers: int = 2,
            high_parallel: int = 4,
            head_hidden: int = 128,
            # 新增参数
            use_routing_regularization: bool = False,
            regularization_lambda: float = 0.01,
            death_threshold: float = 0.01
    ):
        super().__init__()

        # Validate and select scales
        all_scales = ['1m', '3m', '5m', '15m', '30m', '60m', '180m']
        self.scales_to_predict = all_scales[int(start_scale/2):int((end_scale+1)/2)]

        # === Low-freq encoder with improved routing ===
        self.low_freq_encoder = LowFreqDualPathEncoder(
            input_dim=low_input_dim,
            hidden_dim=low_hidden,
            num_experts=num_mid_experts,
            num_layers=low_layers,
            parallelism_factor=low_parallel,
            use_regularization=use_routing_regularization,
            regularization_lambda=regularization_lambda,
            death_threshold=death_threshold
        )

        # === Mid/High unchanged ===
        self.mid_experts = nn.ModuleList([
            MidFreqExpert(
                input_dim=mid_input_dim,
                hidden_dim=mid_hidden,
                num_layers=mid_layers,
                parallelism_factor=mid_parallel
            ) for _ in range(num_mid_experts)
        ])

        self.high_freq_encoder = HighFreqEncoderWithFiLM(
            high_input_dim=high_input_dim,
            mid_context_dim=mid_hidden,
            hidden_dim=high_hidden,
            num_layers=high_layers,
            parallelism_factor=high_parallel
        )

        # === Classification heads ===
        self.classification_heads = MultiScaleClassificationHeads(
            mf_dim=mid_hidden,
            hf_dim=high_hidden,
            class_config=class_config,
            scales=self.scales_to_predict,
            hidden_dim=head_hidden
        )

        # 训练状态
        self.current_epoch = 0

    def forward(
            self,
            x_low: torch.Tensor,
            x_mid: torch.Tensor,
            x_high: torch.Tensor,
            return_regularization: bool = False
    ) -> torch.Tensor:
        # --- Low-freq routing with improved features ---
        expert_weights, reg_loss = self.low_freq_encoder(
            x_low,
            return_regularization=return_regularization
        )

        # --- Mid-freq MoE ---
        expert_outputs = torch.stack([e(x_mid) for e in self.mid_experts], dim=1)  # (B, E, H)
        mf_context = torch.bmm(expert_weights.unsqueeze(1), expert_outputs).squeeze(1)  # (B, H)

        # --- High-freq ---
        hf_seq = self.high_freq_encoder(x_high, mf_context)

        # --- Classification ---
        logits = self.classification_heads(mf_context, hf_seq)

        if return_regularization:
            return logits, reg_loss
        return logits

    def update_epoch(self, epoch: int) -> None:
        """更新epoch，让低频率编码器检查是否需要启用正则化"""
        self.current_epoch = epoch
        self.low_freq_encoder.update_epoch(epoch)

    def get_routing_stats(self, reset: bool = True) -> Dict[str, torch.Tensor]:
        """获取路由统计信息（用于validation阶段）"""
        return self.low_freq_encoder.get_expert_usage_stats(reset=reset)

    def print_routing_stats(self) -> None:
        """打印当前路由统计信息"""
        stats = self.get_routing_stats(reset=False)

        print("\n" + "=" * 60)
        print("路由统计信息:")
        print("=" * 60)
        print(f"专家使用率: {stats['expert_usage'].tolist()}")
        print(f"使用比例: {stats['usage_ratio'].tolist()}")
        print(f"分布熵: {stats['entropy'].item():.4f} (1.0表示完全均匀)")
        print(f"最小使用率: {stats['min_usage'].item():.4f}")
        print(f"最大使用率: {stats['max_usage'].item():.4f}")
        print(f"是否均衡: {stats['is_balanced'].item()}")
        print(f"总样本数: {stats['total_samples'].item()}")
        print("=" * 60)