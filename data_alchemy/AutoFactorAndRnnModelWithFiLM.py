import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
import itertools

from data_alchemy.utils import get_device


class SimpleFiLMLayer(nn.Module):
    """
    最简单的FiLM层：gamma * x + beta
    输入低频上下文，生成gamma和beta，调制特征
    """

    def __init__(
            self,
            feature_dim: int,  # 要调制的特征维度
            context_dim: int,  # 低频上下文维度
            use_bias: bool = True  # 是否使用偏置
    ):
        super().__init__()

        # 简单的线性层生成gamma和beta
        self.gamma_proj = nn.Linear(context_dim, feature_dim)
        self.beta_proj = nn.Linear(context_dim, feature_dim)

        # 初始化：gamma接近1，beta接近0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        if use_bias:
            nn.init.zeros_(self.gamma_proj.bias)
            nn.init.zeros_(self.beta_proj.bias)

    def forward(
            self,
            features: torch.Tensor,  # (B, L, D) 要调制的特征
            lf_context: torch.Tensor  # (B, C) 低频上下文
    ) -> torch.Tensor:
        """
        最简单的FiLM调制：gamma * x + beta

        Args:
            features: 特征序列 (batch_size, seq_len, feature_dim)
            lf_context: 低频上下文 (batch_size, context_dim)

        Returns:
            调制后的特征 (batch_size, seq_len, feature_dim)
        """
        # print(f"SimpleFiLMLayer forward called")
        # 1. 生成gamma和beta
        # gamma_proj: (C) → (D)
        # beta_proj: (C) → (D)
        gamma = self.gamma_proj(lf_context)  # (B, D)
        beta = self.beta_proj(lf_context)  # (B, D)

        # 2. 扩展维度用于广播
        # (B, D) → (B, 1, D)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 3. 调制：gamma * x + beta
        # features: (B, L, D)
        # gamma: (B, 1, D) → 广播到 (B, L, D)
        # beta: (B, 1, D) → 广播到 (B, L, D)
        modulated = gamma * features + beta

        return modulated


class ParallelDNNLayer(nn.Module):
    """
    并行DNN层：多个DNN路径并行计算
    """

    def __init__(
            self,
            name,
            input_dim: int,
            hidden_dims_list: List[List[int]],  # 每个路径的隐藏层维度长度列表
            hidden_dims_nums: List[List[int]],  # 每个路径每个隐藏层的重复数，即每一个长度的维度需要重复多少次
            activations: Union[str, List[str]],
            leaky_alpha: Union[float, None],
            dropouts: Union[float, List[float]],
            use_residual: bool,
            use_layer_norms: Union[bool, List[bool]] = True,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max'
            device: str = get_device()
    ):
        super().__init__()
        self.name = name
        self.num_paths = len(hidden_dims_list)
        self.input_dim = input_dim
        self.aggregation_method = aggregation_method

        # 确保参数列表长度一致
        if isinstance(activations, str):
            activations = [activations] * self.num_paths
        if isinstance(dropouts, float):
            dropouts = [dropouts] * self.num_paths
        if isinstance(use_layer_norms, bool):
            use_layer_norms = [use_layer_norms] * self.num_paths

        assert len(activations) == len(dropouts) == len(use_layer_norms) == self.num_paths

        # 创建每个DNN路径
        self.dnn_paths = nn.ModuleList()
        self.output_dims = []

        for i in range(self.num_paths):
            dnn = LinearDNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims_list[i],
                hidden_dims_nums=hidden_dims_nums[i],
                activation=activations if isinstance(activations, str) else activations[i],
                leaky_alpha=leaky_alpha,
                dropout=dropouts[i] if isinstance(dropouts, list) else dropouts,
                use_layer_norm=use_layer_norms[i] if isinstance(use_layer_norms, list) else use_layer_norms,
                use_residual=use_residual,
            )
            self.dnn_paths.append(dnn)
            self.output_dims.append(hidden_dims_list[i][-1])

        # 聚合层
        if aggregation_method == 'concat':
            self.aggregation_dim = sum(self.output_dims)
        elif aggregation_method in ['sum', 'mean', 'max']:
            self.aggregation_dim = self.output_dims[0]  # 所有路径输出维度必须相同
            for dim in self.output_dims[1:]:
                assert dim == self.aggregation_dim, \
                    f"For {aggregation_method} aggregation, all paths must have same output dim"
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")
        if device == 'auto':
            device = get_device()
        self.to(device)

    def forward(self, x):
        # print(f"{self.name} forwarded ")
        # 并行计算所有路径
        outputs = []
        for dnn_path in self.dnn_paths:
            outputs.append(dnn_path(x))

        # 聚合
        if self.aggregation_method == 'concat':
            return torch.cat(outputs, dim=-1)
        elif self.aggregation_method == 'sum':
            return torch.stack(outputs, dim=0).sum(dim=0)
        elif self.aggregation_method == 'mean':
            return torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            return torch.stack(outputs, dim=0).max(dim=0)[0]
        else:
            return outputs[0]  # 默认返回第一个


# ==================== Linear DNN ====================
class LinearDNN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            hidden_dims_nums,
            activation,
            leaky_alpha,
            dropout: Union[float, List[float]] = 0.1,
            use_layer_norm=True,
            use_residual=True
    ):
        super().__init__()
        self.use_residual = use_residual
        layers = []
        current_dim = input_dim
        # 在 LinearDNN.__init__ 中，替换原 dropout 处理逻辑
        if isinstance(dropout, list):
            assert len(dropout) == len(hidden_dims), \
                f"dropout list length {len(dropout)} != hidden_dims length {len(hidden_dims)}"
            block_dropouts = dropout
        else:
            block_dropouts = [dropout] * len(hidden_dims)
        # 在 LinearDNN.__init__ 开头
        if hidden_dims_nums is None or len(hidden_dims_nums) == 0:
            hidden_dims_nums = [1] * len(hidden_dims)
        assert len(hidden_dims) == len(hidden_dims_nums), \
            "hidden_dims and hidden_dims_nums must have same length"

        for i, (h, num_repeat) in enumerate(zip(hidden_dims, hidden_dims_nums)):
            # 构建一个 block（num_repeat 层）
            block_layers = []
            block_input_dim = current_dim
            dr_block = block_dropouts[i]  # ← 每个 block 一个 dropout 值

            for r in range(num_repeat):
                # Linear
                linear = nn.Linear(current_dim, h)
                if activation == 'leaky_relu':
                    nn.init.kaiming_normal_(linear.weight, a= leaky_alpha,  nonlinearity=activation)
                else:
                    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
                nn.init.zeros_(linear.bias)
                block_layers.append(linear)
                current_dim = h

                # LayerNorm（非全局最后一层）
                is_last_layer = (i == len(hidden_dims) - 1) and (r == num_repeat - 1)
                if use_layer_norm and not is_last_layer:
                    block_layers.append(nn.LayerNorm(h))

                # Activation
                act_map = {
                    'relu': nn.ReLU(),
                    'tanh': nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'leaky_relu': nn.LeakyReLU(leaky_alpha),
                    'elu': nn.ELU()
                }
                if activation not in act_map:
                    raise ValueError(f"Unsupported activation: {activation}")
                block_layers.append(act_map[activation])

                # Dropout
                dr = block_dropouts[i]
                if is_last_layer:
                    dr = dr / 2
                if dr > 0:
                    block_layers.append(nn.Dropout(dr))

            # 包装为残差块（如果启用且重复≥2）
            if self.use_residual and num_repeat >= 2:
                if block_input_dim == h:
                    # 同维：直接残差
                    layers.append(ResidualBlock(nn.Sequential(*block_layers)))
                else:
                    # 异维：投影残差
                    layers.append(ResidualBlock(
                        nn.Sequential(*block_layers),
                        projection=nn.Linear(block_input_dim, h)
                    ))
            else:
                layers.extend(block_layers)

        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        # print(f"LinearDNN called!")
        input_shape = x.shape
        if len(input_shape) == 3:
            B, L, C = input_shape
            x = x.view(-1, C)
            x = self.net(x)
            return x.view(B, L, self.output_dim)
        else:
            return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, main_path, projection=None):
        super().__init__()
        self.main_path = main_path
        self.projection = projection

    def forward(self, x):
        # print("ResidualBlock forwarded")
        residual = x
        out = self.main_path(x)
        if self.projection is not None:
            residual = self.projection(residual)
        return out + residual


class ParallelCausalCNNBlock(nn.Module):
    """
    并行Causal CNN块：多个CNN路径并行计算
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            hidden_dims_list: List[List[int]],  # 每个路径的隐藏层维度列表
            kernel_sizes_list: List[List[int]],  # 每个路径的核大小列表
            dilations_list: Optional[List[List[int]]] = None,
            strides_list: Optional[List[List[int]]] = None,
            use_skip_connections_list: Union[bool, List[bool]] = True,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max'
            device: str = 'auto'
    ):
        super().__init__()
        self.name = name
        self.num_paths = len(hidden_dims_list)
        self.input_dim = input_dim
        self.aggregation_method = aggregation_method

        # 确保参数列表长度一致
        if isinstance(use_skip_connections_list, bool):
            use_skip_connections_list = [use_skip_connections_list] * self.num_paths

        # 设置默认值
        if dilations_list is None:
            dilations_list = [[1] * len(dims) for dims in hidden_dims_list]
        if strides_list is None:
            strides_list = [[1] * len(dims) for dims in hidden_dims_list]

        assert len(kernel_sizes_list) == len(dilations_list) == len(strides_list) == len(
            use_skip_connections_list) == self.num_paths

        # 创建每个CNN路径
        self.cnn_paths = nn.ModuleList()
        self.output_dims = []

        for i in range(self.num_paths):
            cnn = CausalCNNBlock(
                input_dim=input_dim,
                hidden_dims=hidden_dims_list[i],
                kernel_sizes=kernel_sizes_list[i],
                dilations=dilations_list[i],
                strides=strides_list[i],
                use_skip_connections=use_skip_connections_list[i]
            )
            self.cnn_paths.append(cnn)
            self.output_dims.append(hidden_dims_list[i][-1])

        # 聚合层
        if aggregation_method == 'concat':
            self.aggregation_dim = sum(self.output_dims)
        elif aggregation_method in ['sum', 'mean', 'max']:
            self.aggregation_dim = self.output_dims[0]  # 所有路径输出维度必须相同
            for dim in self.output_dims[1:]:
                assert dim == self.aggregation_dim, \
                    f"For {aggregation_method} aggregation, all paths must have same output dim"
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

        self.to(device)

    def calculate_output_length(self, input_len: int) -> int:
        """计算输出长度（所有路径应该相同）"""
        return self.cnn_paths[0].calculate_output_length(input_len)

    def forward(self, x):
        # print(f'{self.name} forwared')
        # 并行计算所有路径
        outputs = []
        for cnn_path in self.cnn_paths:
            outputs.append(cnn_path(x))

        # 聚合
        if self.aggregation_method == 'concat':
            return torch.cat(outputs, dim=-1)
        elif self.aggregation_method == 'sum':
            return torch.stack(outputs, dim=0).sum(dim=0)
        elif self.aggregation_method == 'mean':
            return torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            return torch.stack(outputs, dim=0).max(dim=0)[0]
        else:
            return outputs[0]  # 默认返回第一个


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List


class ParallelRNNLayer(nn.Module):
    """
    并行RNN层：多个RNN路径并行计算（仅单向，用于时间序列预测）
    新增：支持全局残差连接（skip from input to final output）
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            rnn_types: Union[str, List[str]] = 'gru',
            hidden_dims: Union[int, List[int]] = 256,
            num_layers_list: Union[int, List[int]] = 2,
            dropout_list: Union[float, List[float]] = 0.3,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max', 'attention'
            use_attention_aggregation: bool = False,
            use_residual: bool = True,  # ← 新增：是否启用全局残差连接
            device: str = 'auto'
    ):
        super().__init__()
        self.name = name
        # 确保参数列表长度一致
        if isinstance(rnn_types, str):
            rnn_types = [rnn_types]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if isinstance(num_layers_list, int):
            num_layers_list = [num_layers_list]
        if isinstance(dropout_list, float):
            dropout_list = [dropout_list]

        self.num_paths = len(rnn_types)
        self.input_dim = input_dim
        self.aggregation_method = aggregation_method
        self.use_attention_aggregation = use_attention_aggregation
        self.use_residual = use_residual  # ← 保存开关

        # 扩展列表长度
        def extend_list(lst, target_len):
            if len(lst) == 1:
                return lst * target_len
            assert len(lst) == target_len, f"Length mismatch in config lists"
            return lst

        rnn_types = extend_list(rnn_types, self.num_paths)
        hidden_dims = extend_list(hidden_dims, self.num_paths)
        num_layers_list = extend_list(num_layers_list, self.num_paths)
        dropout_list = extend_list(dropout_list, self.num_paths)

        # 创建每个RNN路径（只使用单向RNN）
        rnn_map = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}
        self.rnn_paths = nn.ModuleList()
        self.output_dims = []

        for i in range(self.num_paths):
            rnn_class = rnn_map.get(rnn_types[i].lower(), nn.GRU)
            rnn = rnn_class(
                input_size=input_dim,
                hidden_size=hidden_dims[i],
                num_layers=num_layers_list[i],
                batch_first=True,
                bidirectional=False,
                dropout=dropout_list[i] if num_layers_list[i] > 1 else 0.0
            )
            self.rnn_paths.append(rnn)
            self.output_dims.append(hidden_dims[i])

        # 注意力聚合（如果需要）
        if use_attention_aggregation:
            # 假设所有路径输出维度相同（已在下面检查）
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[0], 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # 聚合层维度计算
        if aggregation_method == 'concat':
            self.aggregation_dim = sum(self.output_dims)
        elif aggregation_method in ['sum', 'mean', 'max', 'attention']:
            self.aggregation_dim = self.output_dims[0]
            for dim in self.output_dims[1:]:
                assert dim == self.aggregation_dim, \
                    f"For {aggregation_method} aggregation, all paths must have same output dim"
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

        # ====== 新增：全局残差连接（从输入直接跳到输出） ======
        if self.use_residual:
            if self.input_dim == self.aggregation_dim:
                self.residual_proj = None  # 直接相加
            else:
                self.residual_proj = nn.Linear(self.input_dim, self.aggregation_dim)
                nn.init.kaiming_normal_(self.residual_proj.weight, nonlinearity='relu')
                nn.init.zeros_(self.residual_proj.bias)
        else:
            self.residual_proj = None
        # ======================================================

        # 设备处理
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.to(device)

    def forward(self, x):
        # print(f"{self.name} forwarded")
        # x: (B, L, input_dim)
        B, L, _ = x.shape

        # 并行计算所有RNN路径
        outputs = []
        hidden_states = []

        for rnn_path in self.rnn_paths:
            rnn_out, hidden_state = rnn_path(x)
            outputs.append(rnn_out)
            hidden_states.append(hidden_state)

        # 聚合 RNN 输出
        if self.aggregation_method == 'concat':
            aggregated = torch.cat(outputs, dim=-1)  # (B, L, agg_dim)
        elif self.aggregation_method == 'sum':
            aggregated = torch.stack(outputs, dim=0).sum(dim=0)
        elif self.aggregation_method == 'mean':
            aggregated = torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            aggregated = torch.stack(outputs, dim=0).max(dim=0)[0]
        elif self.aggregation_method == 'attention' and self.use_attention_aggregation:
            weights = []
            for output in outputs:
                last_output = output[:, -1, :]  # (B, D)
                weight = self.attention(last_output)  # (B, 1)
                weights.append(weight)
            weights = torch.stack(weights, dim=1)  # (B, num_paths, 1)
            weights = F.softmax(weights, dim=1)
            stacked = torch.stack(outputs, dim=0)  # (num_paths, B, L, D)
            weights_expanded = weights.unsqueeze(2).unsqueeze(3)  # (B, num_paths, 1, 1)
            aggregated = (stacked.permute(1, 0, 2, 3) * weights_expanded).sum(dim=1)
        else:
            aggregated = outputs[0]

        # ====== 新增：全局残差连接 ======
        if self.use_residual:
            if self.residual_proj is not None:
                # 投影原始输入以匹配输出维度
                # 注意：x 是 (B, L, input_dim)，需逐时间步投影
                residual = self.residual_proj(x)  # (B, L, aggregation_dim)
            else:
                residual = x  # (B, L, aggregation_dim)，要求 input_dim == aggregation_dim
            aggregated = aggregated + residual
        # ===============================

        return aggregated, hidden_states


class AdvancedDNNCausalCNNRNNParallelWithFiLM(nn.Module):
    """
    增强版：每个模块都支持多个并行路径（专为时间序列预测优化）

    特点：
    1. 所有RNN都是单向的（保持因果性）
    2. 支持残差连接加速收敛
    3. 灵活的并行路径配置
    4. 多种聚合方式
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            input_seq_len: int,
            dnn1: Dict,
            cnn: Dict,
            dnn2: Dict,
            rnn: Dict,
            out_puts: List[int],
            # 残差连接配置（加速收敛）
            use_residual_connections: bool = False,
            residual_strength: float = 0.1,  # 残差连接强度

            # ===== 新增：FiLM调制配置 =====
            use_film: bool = False,  # 是否使用FiLM
            context_dim: Optional[int] = None,  # 低频上下文维度
            init_seed: Optional[int] = None,  # 新增：初始化种子
            deterministic_init: bool = False,  # 新增：确定性初始化
            device: str = 'auto',
            return_sequences=False
    ):
        super().__init__()
        # 设置初始化种子
        if init_seed is not None:
            self._init_with_seed(init_seed, deterministic_init)
        self.name = name
        self.input_dim = input_dim
        self.input_seq_len = input_seq_len
        self.use_residual_connections = use_residual_connections
        self.residual_strength = residual_strength
        if device == 'auto':
            device = get_device()
        self.device = device
        self.use_film = use_film
        self.context_dim = context_dim

        current_seq_len = input_seq_len
        self.seq_len_history = [('input', current_seq_len)]

        if dnn1.get("use_residual") is None:
            dnn1["use_residual"] = use_residual_connections

        # ===== DNN1 (并行多路径) =====
        self.dnn1 = ParallelDNNLayer(
            input_dim=input_dim,
            device=device,
            **dnn1
        )
        dnn1_out_dim = self.dnn1.aggregation_dim

        # ===== CNN (并行多路径) =====
        self.cnn = ParallelCausalCNNBlock(
            input_dim=dnn1_out_dim,
            device=device,
            **cnn,
        )

        # 计算CNN后的序列长度
        current_seq_len = self.cnn.calculate_output_length(current_seq_len)
        self.seq_len_history.append(('cnn', current_seq_len))
        self.cnn_output_len = current_seq_len

        if dnn2.get("use_residual") is None:
            dnn2["use_residual"] = use_residual_connections

        # ===== DNN2 (并行多路径) =====
        self.dnn2 = ParallelDNNLayer(
            input_dim=self.cnn.aggregation_dim,
            device=device,
            **dnn2,
        )
        dnn2_out_dim = self.dnn2.aggregation_dim

        # ===== FiLM调制层（新增） =====
        if use_film:
            if context_dim is None:
                raise ValueError("context_dim must be provided when use_film=True")

            # 创建FiLM层：调制CNN输出特征
            self.film_layer = SimpleFiLMLayer(
                feature_dim=dnn2_out_dim,
                context_dim=context_dim
            )

        # ===== RNN (并行多路径，仅单向) =====
        self.rnn = ParallelRNNLayer(
            input_dim=dnn2_out_dim,
            device=device,
            **rnn,
        )
        rnn_out_dim = self.rnn.aggregation_dim
        self.final_out_dim = out_puts[-1]
        # ===== 构建输出头（在 __init__ 中一次性创建）=====
        self.out_puts = out_puts
        self.return_sequences = return_sequences

        output_layers = []
        in_dim = rnn_out_dim
        for i, out_dim in enumerate(out_puts):
            output_layers.append(nn.Linear(in_dim, out_dim))
            nn.init.kaiming_normal_(output_layers[-1].weight, nonlinearity='relu')
            nn.init.zeros_(output_layers[-1].bias)
            if i < len(out_puts) - 1:
                output_layers.append(nn.LeakyReLU(0.01))
                output_layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        self.output_head = nn.Sequential(*output_layers)
        # 模块间残差连接（加速收敛）
        if use_residual_connections:
            # DNN1到CNN的残差连接投影
            self.dnn1_to_cnn_out_residual = self._create_residual_projection(
                dnn1_out_dim, self.cnn.aggregation_dim, "dnn1_to_cnn"
            )
            # CNN到DNN2的残差连接投影
            self.cnn_to_dnn2_out_residual = self._create_residual_projection(
                self.cnn.aggregation_dim, dnn2_out_dim, "cnn_to_dnn2"
            )
            # DNN2到RNN的残差连接投影
            self.dnn2_to_rnn_out_residual = self._create_residual_projection(
                dnn2_out_dim, rnn_out_dim, "dnn2_to_rnn"
            )
            # RNN -> OUT
            self.rnn_to_end_out_residual = self._create_residual_projection(
                rnn_out_dim, out_puts[-1], "rnn_to_out"
            )

        self.to(device)

    def _create_residual_projection(self, in_dim, out_dim, name):
        """创建残差连接投影层"""
        if in_dim == out_dim:
            return nn.Identity()
        else:
            # 使用带激活函数的投影，更稳定
            proj = nn.Linear(in_dim, out_dim)
            # 残差投影初始化为接近0，这样初始阶段主要依赖主路径
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.zeros_(proj.bias)
            return proj

    def _init_with_seed(self, seed: int, deterministic: bool = False):
        """使用指定种子初始化权重"""
        import random
        import numpy as np
        import torch

        # 保存当前随机状态
        self._saved_random_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
        }

        if torch.cuda.is_available():
            self._saved_random_state['torch_cuda'] = torch.cuda.get_rng_state_all()

        # 设置种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        print(f"Model initialization seed set to {seed}")

    def restore_random_state(self):
        """恢复随机状态"""
        if hasattr(self, '_saved_random_state'):
            import random
            import numpy as np
            import torch

            random.setstate(self._saved_random_state['python'])
            np.random.set_state(self._saved_random_state['numpy'])
            torch.random.set_rng_state(self._saved_random_state['torch'])

            if 'torch_cuda' in self._saved_random_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self._saved_random_state['torch_cuda'])
            print("Random state restored")

    def forward(self, x, return_features=False, lf_context: Optional[torch.Tensor] = None):
        # print(f"{self.name} forward called")
        features = {}
        # DNN1
        dnn1_out = self.dnn1(x)
        if return_features: features['dnn1'] = dnn1_out

        # CNN (带可能的残差连接)
        cnn_input = dnn1_out

        cnn_out = self.cnn(cnn_input)
        if return_features: features['cnn'] = cnn_out

        if self.use_residual_connections:
            residual = self.dnn1_to_cnn_out_residual(cnn_input)
            cnn_out = cnn_out + self.residual_strength * residual

        # DNN2 (带可能的残差连接)
        dnn2_input = cnn_out

        dnn2_out = self.dnn2(dnn2_input)
        if return_features: features['dnn2'] = dnn2_out

        if self.use_residual_connections:
            residual = self.cnn_to_dnn2_out_residual(dnn2_input)
            dnn2_out = dnn2_out + self.residual_strength * residual

        # 3. FiLM调制（可选）
        if self.use_film:
            if lf_context is None:
                raise ValueError("lf_context is required when use_film=True")
            # 简单调制：gamma * dnn2_out + beta
            dnn2_out = self.film_layer(dnn2_out, lf_context)
            # 输出形状不变：(B, L_cnn, D_dnn2)

        # RNN (带可能的残差连接)
        rnn_input = dnn2_out

        rnn_out, rnn_hidden = self.rnn(rnn_input)

        if self.use_residual_connections:
            residual = self.dnn2_to_rnn_out_residual(rnn_input)
            rnn_out = rnn_out + self.residual_strength * residual

        if return_features:
            features['rnn_out'] = rnn_out
            features['rnn_hidden'] = rnn_hidden

        # ===== 输出头前向 =====
        if self.return_sequences:
            # (B, L, D) -> (B*L, D) -> (B*L, final) -> (B, L, final)
            B, L, D = rnn_out.shape
            outlayer_input = rnn_out.view(-1, D)
            out_flat = self.output_head(outlayer_input)
            out = out_flat.view(B, L, -1)
        else:
            # 取最后一个时间步
            outlayer_input = rnn_out[:, -1, :]
            out = self.output_head(outlayer_input)  # (B, final_out_dim)
        if self.use_residual_connections:
            residual = self.rnn_to_end_out_residual(outlayer_input)
            if self.return_sequences:
                residual = residual.view(B, L, -1)
            out = out + self.residual_strength * residual
        return (out, features) if return_features else out

    def get_seq_len_info(self):
        return {
            'input_seq_len': self.input_seq_len,
            'output_seq_len_after_cnn': self.cnn_output_len,
            'history': self.seq_len_history,
            'total_reduction': self.input_seq_len - self.cnn_output_len,
            'dnn1_output_dim': self.dnn1.aggregation_dim,
            'cnn_output_dim': self.cnn.aggregation_dim,
            'dnn2_output_dim': self.dnn2.aggregation_dim,
            'rnn_output_dim': self.rnn.aggregation_dim,
            'use_residual_connections': self.use_residual_connections
        }


# ==================== Causal CNN Block ====================

class CausalCNNBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            kernel_sizes: List[int],
            dilations: List[int],
            strides: List[int],
            use_skip_connections: bool = True
    ):
        super().__init__()

        assert len(hidden_dims) == len(kernel_sizes) == len(dilations) == len(strides)

        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.strides = strides

        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList() if use_skip_connections else None

        in_ch = input_dim
        for i, out_ch in enumerate(hidden_dims):
            k = kernel_sizes[i]
            d = dilations[i]
            s = strides[i]

            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=0,  # ← 关键：无 padding
                dilation=d
            )
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            nn.init.zeros_(conv.bias)

            layer = nn.Sequential(
                conv,
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)

            # Skip connection projection
            if use_skip_connections:
                if in_ch == out_ch and s == 1:
                    self.skip_projs.append(nn.Identity())
                else:
                    self.skip_projs.append(nn.Conv1d(in_ch, out_ch, 1, stride=s))
            else:
                self.skip_projs.append(None)

            in_ch = out_ch

        self.output_dim = hidden_dims[-1]

    def calculate_output_length(self, input_len: int) -> int:
        """计算经过整个 Causal CNN block 后的输出长度"""
        L = input_len
        for k, d, s in zip(self.kernel_sizes, self.dilations, self.strides):
            # 因果卷积实际等效于：先 pad left = (k-1)*d，再做 valid conv
            # 输出长度 = (L + left_pad - effective_kernel) // s + 1
            # effective_kernel = (k - 1) * d + 1
            # left_pad = (k - 1) * d
            # => L_out = (L + (k-1)*d - ((k-1)*d + 1)) // s + 1 = (L - 1) // s + 1
            L = (L - 1) // s + 1
        return L

    def forward(self, x):
        # x: (B, L, C) → (B, C, L)
        # print(f"CausalCNNBlock forward called")
        x = x.transpose(1, 2)

        for i, layer in enumerate(self.layers):
            residual = x
            k = self.kernel_sizes[i]
            d = self.dilations[i]
            left_pad = (k - 1) * d

            # 手动左填充（因果关键！）
            x_padded = F.pad(x, (left_pad, 0))  # (B, C, L + left_pad)

            x = layer(x_padded)

            # Skip connection
            if self.skip_projs[i] is not None:
                # 对 residual 做相同 stride 的下采样
                residual_down = self.skip_projs[i](residual)
                if residual_down.shape == x.shape:
                    x = x + residual_down

        return x.transpose(1, 2)  # (B, C, L) → (B, L, C)


import yaml
import torch
from typing import Dict, Any, Optional
import copy


def create_model_from_yaml(config_path: str,
                           init_seed: Optional[int] = None) -> AdvancedDNNCausalCNNRNNParallelWithFiLM:
    """
    从YAML配置文件创建AdvancedDNNCausalCNNRNNParallelWithFiLM模型

    Args:
        config_path: YAML配置文件路径

    Returns:
        AdvancedDNNCausalCNNRNNParallelWithFiLM实例
        :param config_path:
        :param init_seed:
    """
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取模型配置部分
    model_config = config.get('model', {})

    # 提取种子配置（优先使用函数参数）
    seed_from_config = model_config.get('init_seed')
    final_seed = init_seed if init_seed is not None else seed_from_config

    # 获取确定性配置
    deterministic_init = model_config.get('deterministic_init', False)
    # 基础参数
    base_params = {
        'input_dim': model_config.get('input_dim'),
        'input_seq_len': model_config.get('input_seq_len', 600),
        'output_type': model_config.get('output_type', 'regression'),
        'num_classes': model_config.get('num_classes'),
        'output_dim': model_config.get('output_dim'),
        'return_sequences': model_config.get('return_sequences', False),
        'use_residual_connections': model_config.get('use_residual_connections', False),
        'residual_strength': model_config.get('residual_strength', 0.1),
        'use_film': model_config.get('use_film', False),
        'lf_context_dim': model_config.get('lf_context_dim'),
        'init_seed': final_seed,  # 传递种子参数
        'deterministic_init': deterministic_init,
    }

    # 设备配置
    device = model_config.get('device')
    base_params['device'] = device

    # ===== DNN1配置 =====
    dnn1_config = model_config.get('dnn1', {})
    if dnn1_config:
        base_params.update({
            'dnn1_hidden_dims_list': dnn1_config.get('hidden_dims_list'),
            'dnn1_hidden_dims_nums': dnn1_config.get('hidden_dims_nums'),
            'dnn1_activations': dnn1_config.get('activations', 'leaky_relu'),
            'dnn1_dropouts': dnn1_config.get('dropouts', 0.1),
            'dnn1_use_layer_norms': dnn1_config.get('use_layer_norms', True),
            'dnn1_aggregation_method': dnn1_config.get('aggregation_method', 'concat'),
        })

    # ===== CNN配置 =====
    cnn_config = model_config.get('cnn', {})
    if cnn_config:
        base_params.update({
            'cnn_hidden_dims_list': cnn_config.get('hidden_dims_list', [[128, 256, 512]]),
            'cnn_kernel_sizes_list': cnn_config.get('kernel_sizes_list', [[3, 5, 7]]),
            'cnn_dilations_list': cnn_config.get('dilations_list'),
            'cnn_strides_list': cnn_config.get('strides_list'),
            'cnn_use_skip_connections_list': cnn_config.get('use_skip_connections_list', True),
            'cnn_aggregation_method': cnn_config.get('aggregation_method', 'concat'),
        })

    # ===== DNN2配置 =====
    dnn2_config = model_config.get('dnn2', {})
    if dnn2_config:
        base_params.update({
            'dnn2_hidden_dims_list': dnn2_config.get('hidden_dims_list', [[256, 128]]),
            'dnn2_hidden_dims_nums': dnn2_config.get('dnn_hidden_dims_nums', []),
            'dnn2_activations': dnn2_config.get('activations', 'leaky_relu'),
            'dnn2_dropouts': dnn2_config.get('dropouts', 0.2),
            'dnn2_use_layer_norms': dnn2_config.get('use_layer_norms', True),
            'dnn2_aggregation_method': dnn2_config.get('aggregation_method', 'concat'),
        })

    # ===== RNN配置 =====
    rnn_config = model_config.get('rnn', {})
    if rnn_config:
        base_params.update({
            'rnn_types': rnn_config.get('types', 'gru'),
            'rnn_hidden_dims': rnn_config.get('hidden_dims', 256),
            'rnn_num_layers_list': rnn_config.get('num_layers_list', 2),
            'rnn_dropout_list': rnn_config.get('dropout_list', 0.3),
            'rnn_aggregation_method': rnn_config.get('aggregation_method', 'concat'),
            'rnn_use_attention_aggregation': rnn_config.get('use_attention_aggregation', False),
        })

    # 验证必填参数
    if base_params['input_dim'] is None:
        raise ValueError("input_dim must be specified in config")

    if base_params['use_film'] and base_params['lf_context_dim'] is None:
        raise ValueError("lf_context_dim must be specified when use_film=True")

    if base_params['output_type'] == 'classification' and base_params['num_classes'] is None:
        raise ValueError("num_classes must be specified for classification output_type")

        # 设置全局随机种子（如果需要）
    if final_seed is not None:
        _set_global_seed_for_init(final_seed, deterministic_init)

    # 创建模型
    model = AdvancedDNNCausalCNNRNNParallelWithFiLM(**base_params)

    # 打印模型信息
    print(f"Model created from config: {config_path}")
    print(f"  Input dim: {base_params['input_dim']}")
    print(f"  Input seq len: {base_params['input_seq_len']}")
    print(f"  Output type: {base_params['output_type']}")
    print(f"  Use FiLM: {base_params['use_film']}")
    print(f"  Use residual connections: {base_params['use_residual_connections']}")

    seq_info = model.get_seq_len_info()
    print(f"  Sequence length after CNN: {seq_info['output_seq_len_after_cnn']}")
    print(f"  Total reduction: {seq_info['total_reduction']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def _set_global_seed_for_init(seed: int, deterministic: bool = False):
    """为模型初始化设置全局种子"""
    import random
    import numpy as np
    import torch

    print(f"Setting global seed for model initialization: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 注意：不要在这里禁用 cuDNN，因为会影响后续训练


def create_model_from_dict(config_dict: Dict[str, Any]) -> AdvancedDNNCausalCNNRNNParallelWithFiLM:
    """
    从字典配置创建模型

    Args:
        config_dict: 配置字典

    Returns:
        AdvancedDNNCausalCNNRNNParallelWithFiLM实例
    """
    # 直接使用字典参数
    return AdvancedDNNCausalCNNRNNParallelWithFiLM(**config_dict)
