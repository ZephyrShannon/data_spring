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

    def print_gradient_analysis(self, logger, name):
        for i, dnn_layer in enumerate(self.dnn_paths):
            name = f"{name}[{i}]"
            dnn_layer.print_gradient_analysis(logger, name)

    def forward(self, x):
        # print(f"{self.name} forwarded ")
        # 并行计算所有路径
        outputs = []
        for dnn_path in self.dnn_paths:
            outputs.append(dnn_path(x))
        if self.num_paths > 1:
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
        else:
            return outputs[0]


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, main_path, projection=None, strength=0.1):
        super().__init__()
        self.main_path = main_path
        self.projection = projection
        self.strength = strength

    def forward(self, x):
        out = self.main_path(x)
        residual = self.projection(x)
        # 残差连接，strength控制残差强度
        return out + self.strength * residual

    def get_gradient(self):
        """
        获取残差块内主路径和残差路径的梯度

        Returns:
            dict: {
                'main_path': float,  # 主路径的平均梯度norm
                'residual_path': float,  # 残差路径的平均梯度norm
                'strength': float  # 残差强度
            }
        """
        result = {
            'main_path': 0.0,
            'residual_path': 0.0,
            'strength': self.strength
        }

        # 收集主路径的梯度（main_path中的所有参数）
        main_grads = []
        for name, param in self.main_path.named_parameters():
            if param.grad is not None:
                main_grads.append(param.grad.norm().item())

        if main_grads:
            result['main_path'] = sum(main_grads) / len(main_grads)

        # 收集投影层的梯度
        if self.projection is not None:
            proj_grads = []
            for name, param in self.projection.named_parameters():
                if param.grad is not None:
                    proj_grads.append(param.grad.norm().item())
            if proj_grads:
                result['residual_path'] = sum(proj_grads) / len(proj_grads)
            else:
                result['residual_path'] = self.strength * result['main_path']

        return result


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
            use_residual=True,
            residual_frequency: int = 2,  # 每多少层添加一次残差
            residual_strength: float = 0.1
    ):
        super().__init__()
        self.use_residual = use_residual
        self.residual_frequency = residual_frequency
        self.residual_strength = residual_strength
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.hidden_dims_nums = hidden_dims_nums

        # 用于存储残差块信息，方便梯度分析
        self.residual_blocks = nn.ModuleList()

        # 构建基础层（不带残差）
        base_layers = self._build_base_layers(
            input_dim, hidden_dims, hidden_dims_nums,
            activation, leaky_alpha, dropout, use_layer_norm
        )

        # 根据频率添加残差
        if use_residual and residual_frequency > 0:
            self.net, self.residual_blocks = self._add_residual_connections(
                base_layers, residual_frequency, residual_strength
            )
        else:
            self.net = nn.Sequential(*[layer for _, layer, _, _ in base_layers])
            self.residual_blocks = nn.ModuleList()

        self.output_dim = hidden_dims[-1]

    def _build_base_layers(self, input_dim, hidden_dims, hidden_dims_nums,
                           activation, leaky_alpha, dropout, use_layer_norm):
        """构建基础层（不带残差）"""
        layers = []
        current_dim = input_dim

        # 处理dropout
        if isinstance(dropout, list):
            block_dropouts = dropout
        else:
            block_dropouts = [dropout] * len(hidden_dims)

        # 激活函数映射
        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(leaky_alpha),
            'elu': nn.ELU()
        }

        for i, (h, num_repeat) in enumerate(zip(hidden_dims, hidden_dims_nums)):
            for r in range(num_repeat):
                is_last = (i == len(hidden_dims) - 1) and (r == num_repeat - 1)

                # Linear
                linear = nn.Linear(current_dim, h)
                if activation == 'leaky_relu':
                    nn.init.kaiming_normal_(linear.weight, a=leaky_alpha, nonlinearity='leaky_relu')
                else:
                    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
                nn.init.zeros_(linear.bias)
                layers.append(('linear', linear, current_dim, h))
                current_dim = h

                # LayerNorm
                if use_layer_norm and not is_last:
                    norm = nn.LayerNorm(h)
                    layers.append(('norm', norm, h, h))

                # Activation
                layers.append(('act', act_map[activation], h, h))

                # Dropout
                dr = block_dropouts[i]
                if is_last:
                    dr = dr / 2
                if dr > 0:
                    layers.append(('dropout', nn.Dropout(dr), h, h))

        return layers

    def _add_residual_connections(self, base_layers, frequency, strength):
        """根据频率添加残差连接"""
        layers = []
        residual_blocks = nn.ModuleList()
        current_block = []
        linear_count = 0
        block_input_dim = base_layers[0][2] if base_layers else None

        for layer_info in base_layers:
            layer_type, layer, in_dim, out_dim = layer_info
            current_block.append(layer)

            if layer_type == 'linear':
                linear_count += 1

            # 当达到frequency个线性层时，创建一个残差块
            if linear_count >= frequency:
                block_output_dim = out_dim
                # 创建残差块
                if block_input_dim == block_output_dim:
                    # 同维：直接残差
                    block = ResidualBlock(
                        nn.Sequential(*current_block),
                        projection=nn.Identity(),
                        strength=strength
                    )
                else:
                    # 异维：投影残差
                    projection = nn.Linear(block_input_dim, block_output_dim)
                    nn.init.xavier_uniform_(projection.weight, gain=1.0)
                    nn.init.zeros_(projection.bias)
                    block = ResidualBlock(
                        nn.Sequential(*current_block),
                        projection=projection,
                        strength=strength
                    )

                residual_blocks.append(block)
                layers.append(block)
                # 重置
                current_block = []
                linear_count = 0
                block_input_dim = block_output_dim

        # 处理剩余的层
        if current_block:
            layers.extend(current_block)

        return nn.Sequential(*layers), residual_blocks

    def forward(self, x):
        input_shape = x.shape
        if len(input_shape) == 3:
            B, L, C = input_shape
            x = x.view(-1, C)
            x = self.net(x)
            return x.view(B, L, self.output_dim)
        else:
            return self.net(x)

    def get_gradient(self):
        """
        获取第一个和最后一个ResidualBlock的梯度信息

        Returns:
            dict: {
                'first_block': {
                    'main_path': float,
                    'residual_path': float,
                    'projection': float,
                    'strength': float,
                    'grad_ratio': float  # residual_path / main_path
                },
                'last_block': {
                    'main_path': float,
                    'residual_path': float,
                    'projection': float,
                    'strength': float,
                    'grad_ratio': float  # residual_path / main_path
                },
                'grad_decrease': {
                    'main_path': float,  # 最后一个block的main_path / 第一个block的main_path
                    'residual_path': float,  # 最后一个block的residual_path / 第一个block的residual_path
                    'projection': float  # 最后一个block的projection / 第一个block的projection
                }
            }
        """
        result = {
            'first_block': {},
            'last_block': {},
            'grad_decrease': {}
        }

        if len(self.residual_blocks) == 0:
            # 如果没有残差块，返回空结果
            return result

        # 获取第一个和最后一个残差块的梯度
        first_block_grad = self.residual_blocks[0].get_gradient()
        last_block_grad = self.residual_blocks[-1].get_gradient()

        # 计算梯度比率（residual_path / main_path）
        first_block_grad['grad_ratio'] = (
            first_block_grad['residual_path'] / first_block_grad['main_path']
            if first_block_grad['main_path'] > 0 else 0
        )
        last_block_grad['grad_ratio'] = (
            last_block_grad['residual_path'] / last_block_grad['main_path']
            if last_block_grad['main_path'] > 0 else 0
        )

        result['first_block'] = first_block_grad
        result['last_block'] = last_block_grad

        # 计算梯度衰减率（最后一个 / 第一个）
        for key in ['main_path', 'residual_path']:
            if first_block_grad[key] > 0 and last_block_grad[key] > 0:
                result['grad_decrease'][key] = last_block_grad[key] / first_block_grad[key]
            else:
                result['grad_decrease'][key] = 0.0

        return result

    def print_gradient_analysis(self, logger, name=""):
        """打印梯度分析结果"""
        grad_info = self.get_gradient()

        logger.info("\n" + "=" * 60)
        logger.info(f"{name} LinearDNN 梯度分析")
        logger.info("=" * 60)

        if not grad_info['first_block']:
            logger.info("没有残差块，无法分析")
            return
        # 第一个残差块
        logger.info(f"  输入: [{grad_info['first_block']['residual_path']:.6f}/{grad_info['first_block']['main_path']:.6f}={grad_info['first_block']['grad_ratio']:.4f}]")
        # 最后一个残差块
        logger.info(
            f"  输出: [{grad_info['last_block']['residual_path']:.6f}/{grad_info['last_block']['main_path']:.6f}={grad_info['last_block']['grad_ratio']:.4f}]")
        # 梯度衰减
        logger.info(f"\n【梯度衰减率 ([主， 残]衰减)]:[{grad_info['grad_decrease']['main_path']:.4f},{grad_info['grad_decrease']['residual_path']:.4f}]")
        # 分析
        logger.info("\n【分析】")
        if 0 < grad_info['grad_decrease']['main_path'] < 1:
            logger.info(f"  ⚠️ 主路径梯度衰减了 {1 / grad_info['grad_decrease']['main_path']:.2f} 倍")
        else:
            logger.info("  ✓ 主路径梯度保持良好")

        if 0 < grad_info['grad_decrease']['residual_path'] < 1:
            logger.info(f"  ⚠️ 残差路径梯度衰减了 {1 / grad_info['grad_decrease']['residual_path']:.2f} 倍")
        else:
            logger.info("  ✓ 残差路径梯度保持良好")
        logger.info("=" * 60)
        return grad_info


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

    def get_gradient_decay(self, logger, name_prefix=""):
        """
        计算ParallelDNNLayer中每个LinearDNN路径的梯度衰减

        衰减率 = 第一个线性层的梯度 / 最后一个线性层的梯度

        Args:
            logger: 日志记录器
            name_prefix: 名称前缀

        Returns:
            dict: 每个路径的梯度衰减信息
        """
        results = {
            'paths': [],
            'summary': {}
        }

        for i, dnn_path in enumerate(self.dnn_paths):
            path_name = f"{name_prefix}_path_{i}"

            # 获取这个LinearDNN的梯度衰减
            path_result = dnn_path.get_gradient_decay(logger, path_name)

            if path_result:
                results['paths'].append({
                    'path_idx': i,
                    'decay_ratio': path_result.get('decay_ratio', 0),
                    'input_grad': path_result.get('first_layer_grad', 0),
                    'output_grad': path_result.get('last_layer_grad', 0)
                })


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
            cnn: Dict = None,
            dnn2: Dict = None,
            rnn: Dict = None,
            out_puts: Dict = None,
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

        if rnn is not None:
            # ===== RNN (并行多路径，仅单向) =====
            self.rnn = ParallelRNNLayer(
                input_dim=dnn2_out_dim,
                device=device,
                **rnn,
            )
            rnn_out_dim = self.rnn.aggregation_dim
        else:
            self.rnn = None
            rnn_out_dim = dnn2_out_dim

        self.return_sequences = return_sequences
        if out_puts is not None:
            if out_puts.get("use_residual") is None:
                out_puts["use_residual"] = use_residual_connections
            self.output_head = ParallelDNNLayer(
                input_dim=rnn_out_dim,
                device=device,
                **out_puts,
            )
            self.final_out_dim = self.output_head.aggregation_dim
        else:
            self.output_head = None
            self.final_out_dim = rnn_out_dim
        # 模块间残差连接（加速收敛）
        if use_residual_connections:
            # DNN1到CNN的残差连接投影
            self.dnn1_to_cnn_out_residual = self._create_residual_projection(
                dnn1_out_dim, self.cnn.aggregation_dim,
            )
            # CNN到DNN2的残差连接投影
            self.cnn_to_dnn2_out_residual = self._create_residual_projection(
                self.cnn.aggregation_dim, dnn2_out_dim,
            )
            if rnn is not None:
                # DNN2到RNN的残差连接投影
                self.dnn2_to_rnn_out_residual = self._create_residual_projection(
                    dnn2_out_dim, rnn_out_dim,
                )
            else:
                self.dnn2_to_rnn_out_residual = None
            # RNN -> OUT
            if out_puts is not None:
                self.rnn_to_end_out_residual = self._create_residual_projection(
                    rnn_out_dim, self.output_head.aggregation_dim,
                )
            else:
                self.rnn_to_end_out_residual = None
        self.to(device)

    def _create_residual_projection(self, in_dim, out_dim):
        """创建残差连接投影层"""
        if in_dim == out_dim:
            return nn.Identity()
        else:
            # 使用带激活函数的投影，更稳定
            proj = nn.Linear(in_dim, out_dim)
            # 残差投影初始化为接近0，这样初始阶段主要依赖主路径
            nn.init.xavier_uniform_(proj.weight, gain=1)
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
        if self.rnn is not None:
            rnn_out, rnn_hidden = self.rnn(rnn_input)
            if return_features:
                features['rnn_out'] = rnn_out
                features['rnn_hidden'] = rnn_hidden
        else:
            rnn_out = rnn_input


        if self.use_residual_connections:
            if self.dnn2_to_rnn_out_residual is not None:
                residual = self.dnn2_to_rnn_out_residual(rnn_input)
                rnn_out = rnn_out + self.residual_strength * residual



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
        ret = {
            'input_seq_len': self.input_seq_len,
            'output_seq_len_after_cnn': self.cnn_output_len,
            'history': self.seq_len_history,
            'total_reduction': self.input_seq_len - self.cnn_output_len,
            'dnn1_output_dim': self.dnn1.aggregation_dim,
            'cnn_output_dim': self.cnn.aggregation_dim,
            'dnn2_output_dim': self.dnn2.aggregation_dim,
            'use_residual_connections': self.use_residual_connections
        }
        if self.rnn is not None:
            ret['rnn_output_dim'] = self.rnn.aggregation_dim,

    def get_all_gradient_decays(self, logger):
        """
        获取模型中所有模块的梯度衰减

        包括: dnn1, cnn, dnn2, rnn, final_out, 以及4个残差连接

        Args:
            logger: 日志记录器

        Returns:
            dict: 各模块的梯度衰减信息
        """
        results = {
            'dnn1': {},
            'cnn': {},
            'dnn2': {},
            'final_out': {},
            'residual_connections': {},
            'summary': {}
        }

        # 1. 分析 dnn1 (ParallelDNNLayer)
        logger.info("\n" + "=" * 70)
        logger.info("开始分析 dnn1 模块")
        logger.info("=" * 70)
        results['dnn1'] = self.dnn1.get_gradient_decay(logger, "dnn1")

        # 2. 分析 cnn (ParallelCausalCNNBlock)
        logger.info("\n" + "=" * 70)
        logger.info("开始分析 cnn 模块")
        logger.info("=" * 70)
        results['cnn'] = self._analyze_cnn_gradients(logger)

        # 3. 分析 dnn2 (ParallelDNNLayer)
        logger.info("\n" + "=" * 70)
        logger.info("开始分析 dnn2 模块")
        logger.info("=" * 70)
        results['dnn2'] = self.dnn2.get_gradient_decay(logger, "dnn2")

        # 4. 分析 rnn (ParallelRNNLayer)
        if self.rnn is not None:
            logger.info("\n" + "=" * 70)
            logger.info("开始分析 rnn 模块")
            logger.info("=" * 70)
            results['rnn'] = self._analyze_rnn_gradients(logger)

        # 5. 分析 final_out (ParallelDNNLayer)
        logger.info("\n" + "=" * 70)
        logger.info("开始分析 final_out 模块")
        logger.info("=" * 70)
        results['final_out'] = self.output_head.get_gradient_decay(logger, "final_out")

        # 6. 分析残差连接
        logger.info("\n" + "=" * 70)
        logger.info("开始分析残差连接")
        logger.info("=" * 70)
        results['residual_connections'] = self._analyze_residual_connections(logger)

        # 7. 汇总分析
        results['summary'] = self._summarize_gradient_analysis(results, logger)

        return results

    def _analyze_cnn_gradients(self, logger):
        """
        分析CNN模块的梯度
        CNN由多个Conv1d层组成
        """
        results = {
            'layers': [],
            'summary': {}
        }

        # 收集所有Conv1d层的梯度
        conv_grads = []
        for name, module in self.cnn.named_modules():
            if isinstance(module, nn.Conv1d):
                if module.weight.grad is not None:
                    grad_norm = module.weight.grad.norm().item()
                    conv_grads.append({
                        'name': name,
                        'grad_norm': grad_norm
                    })

        if conv_grads:
            # 按顺序排序
            first_grad = conv_grads[0]['grad_norm']
            last_grad = conv_grads[-1]['grad_norm']
            decay_ratio = first_grad / last_grad if last_grad > 0 else 0

            results['layers'] = conv_grads
            results['summary'] = {
                'first_layer_grad': first_grad,
                'last_layer_grad': last_grad,
                'decay_ratio': decay_ratio,
                'num_layers': len(conv_grads),
                'is_healthy': 0.01 < decay_ratio < 100
            }

            # 打印详细分析
            logger.info("\nCNN梯度分析:")
            logger.info("-" * 60)
            for i, layer in enumerate(conv_grads):
                logger.info(f"层{i}: {layer['name']} - 梯度norm: {layer['grad_norm']:.6f}")
            logger.info("-" * 60)
            logger.info(f"第一个卷积层梯度: {first_grad:.6f}")
            logger.info(f"最后一个卷积层梯度: {last_grad:.6f}")
            logger.info(f"衰减率: {decay_ratio:.4f}")
            logger.info(f"状态: {'✓健康' if 0.01 < decay_ratio < 100 else '⚠️问题'}")

        return results

    def _analyze_rnn_gradients(self, logger):
        """
        分析RNN模块的梯度
        RNN由多个RNN路径组成
        """
        results = {
            'paths': [],
            'summary': {}
        }

        # 分析每个RNN路径
        for i, rnn_path in enumerate(self.rnn.rnn_paths):
            path_grads = []

            # 收集RNN参数梯度
            for name, param in rnn_path.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    path_grads.append({
                        'name': name,
                        'grad_norm': grad_norm
                    })

            if path_grads:
                # 计算平均梯度
                avg_grad = sum(g['grad_norm'] for g in path_grads) / len(path_grads)
                results['paths'].append({
                    'path_idx': i,
                    'avg_grad': avg_grad,
                    'num_params': len(path_grads)
                })

        if results['paths']:
            # 计算汇总
            avg_grads = [p['avg_grad'] for p in results['paths']]
            results['summary'] = {
                'mean_grad': sum(avg_grads) / len(avg_grads),
                'min_grad': min(avg_grads),
                'max_grad': max(avg_grads),
                'num_paths': len(results['paths'])
            }

            # 打印分析
            logger.info("\nRNN梯度分析:")
            logger.info("-" * 60)
            for path in results['paths']:
                logger.info(f"路径{path['path_idx']}: 平均梯度 {path['avg_grad']:.6f}")
            logger.info("-" * 60)
            logger.info(f"平均梯度: {results['summary']['mean_grad']:.6f}")

        return results

    def _analyze_residual_connections(self, logger):
        """
        分析4个残差连接的梯度
        """
        results = {}

        residual_names = [
            ('dnn1_to_cnn', 'dnn1_to_cnn_out_residual'),
            ('cnn_to_dnn2', 'cnn_to_dnn2_out_residual'),
            ('dnn2_to_rnn', 'dnn2_to_rnn_out_residual'),
            ('rnn_to_out', 'rnn_to_end_out_residual')
        ]

        for name, attr_name in residual_names:
            if hasattr(self, attr_name):
                residual_module = getattr(self, attr_name)

                # 收集残差连接的梯度
                grad_norms = []
                for param_name, param in residual_module.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())

                if grad_norms:
                    avg_grad = sum(grad_norms) / len(grad_norms)
                    results[name] = {
                        'avg_grad': avg_grad,
                        'num_params': len(grad_norms),
                        'type': 'Identity' if isinstance(residual_module, nn.Identity) else 'Linear'
                    }

                    logger.info(f"\n{name} 残差连接:")
                    logger.info(f"  类型: {results[name]['type']}")
                    logger.info(f"  平均梯度: {avg_grad:.6f}")

        return results

    def _summarize_gradient_analysis(self, results, logger):
        """
        汇总梯度分析结果
        """
        summary = {
            'healthy_modules': [],
            'problem_modules': [],
            'recommendations': []
        }

        # 检查每个模块的健康状态
        module_checks = [
            ('dnn1', results['dnn1'].get('summary', {}).get('mean_decay', 0) if 'summary' in results['dnn1'] else 0),
            ('dnn2', results['dnn2'].get('summary', {}).get('mean_decay', 0) if 'summary' in results['dnn2'] else 0),
            ('final_out',
             results['final_out'].get('summary', {}).get('mean_decay', 0) if 'summary' in results['final_out'] else 0),
            ('cnn', results['cnn'].get('summary', {}).get('decay_ratio', 0) if 'summary' in results['cnn'] else 0)
        ]

        logger.info("\n" + "=" * 70)
        logger.info("梯度分析汇总")
        logger.info("=" * 70)

        for module_name, decay in module_checks:
            if decay == 0:
                status = "未知"
            elif 0.01 < decay < 100:
                status = "✓健康"
                summary['healthy_modules'].append(module_name)
            else:
                status = "⚠️问题"
                summary['problem_modules'].append((module_name, decay))

            logger.info(f"{module_name:10}: 衰减率 {decay:8.4f} {status}")

        # RNN特殊处理
        if 'rnn' in results and 'summary' in results['rnn']:
            rnn_grad = results['rnn']['summary'].get('mean_grad', 0)
            logger.info(f"{'rnn':10}: 平均梯度 {rnn_grad:.6f}")

            if rnn_grad < 1e-5:
                logger.info(f"           ⚠️ RNN梯度过小")
                summary['problem_modules'].append(('rnn', rnn_grad))

        # 提供建议
        logger.info("\n【优化建议】")
        if summary['problem_modules']:
            for module, value in summary['problem_modules']:
                if module in ['dnn1', 'dnn2', 'final_out'] and value > 100:
                    logger.info(f"  - {module} 梯度衰减严重，建议增加残差强度")
                elif module == 'cnn' and value > 100:
                    logger.info(f"  - {module} 梯度衰减严重，考虑减少CNN层数")
                elif module == 'rnn':
                    logger.info(f"  - {module} 梯度过小，考虑增加RNN的残差强度")

            logger.info(
                f"  整体建议: 将残差强度从 {self.residual_strength} 增加到 {min(0.5, self.residual_strength * 1.5):.2f}")
        else:
            logger.info("  ✓ 所有模块梯度状态良好")

        return summary

    def print_complete_gradient_analysis(self, logger):
        """
        打印完整的梯度分析报告
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"模型 {self.name} 完整梯度分析报告")
        logger.info("=" * 80)

        # 获取所有梯度信息
        results = self.get_all_gradient_decays(logger)

        # 打印最终汇总
        logger.info("\n" + "=" * 80)
        logger.info("分析完成")
        logger.info("=" * 80)

        return results

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
