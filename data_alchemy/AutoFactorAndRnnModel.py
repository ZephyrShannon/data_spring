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


class ParallelDNNLayer(nn.Module):
    """
    并行DNN层：多个DNN路径并行计算
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims_list: List[List[int]],  # 每个路径的隐藏层维度列表
            activations: Union[str, List[str]] = 'relu',
            dropouts: Union[float, List[float]] = 0.1,
            use_layer_norms: Union[bool, List[bool]] = True,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max'
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

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
                activation=activations[i],
                dropout=dropouts[i],
                use_layer_norm=use_layer_norms[i]
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

        self.to(device)

    def forward(self, x):
        # 并行计算所有路径
        outputs = []
        for dnn_path in self.dnn_paths:
            outputs.append(dnn_path(x))

        # 聚合
        if self.aggregation_method == 'concat':
            return torch.cat(outputs, dim=-1)
        elif self.aggregation_method == 'sum':
            return sum(outputs)
        elif self.aggregation_method == 'mean':
            return torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            return torch.stack(outputs, dim=0).max(dim=0)[0]
        else:
            return outputs[0]  # 默认返回第一个


class ParallelCausalCNNBlock(nn.Module):
    """
    并行Causal CNN块：多个CNN路径并行计算
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims_list: List[List[int]],  # 每个路径的隐藏层维度列表
            kernel_sizes_list: List[List[int]],  # 每个路径的核大小列表
            dilations_list: Optional[List[List[int]]] = None,
            strides_list: Optional[List[List[int]]] = None,
            use_skip_connections_list: Union[bool, List[bool]] = True,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max'
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

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
        # 并行计算所有路径
        outputs = []
        for cnn_path in self.cnn_paths:
            outputs.append(cnn_path(x))

        # 聚合
        if self.aggregation_method == 'concat':
            return torch.cat(outputs, dim=-1)
        elif self.aggregation_method == 'sum':
            return sum(outputs)
        elif self.aggregation_method == 'mean':
            return torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            return torch.stack(outputs, dim=0).max(dim=0)[0]
        else:
            return outputs[0]  # 默认返回第一个


class ParallelRNNLayer(nn.Module):
    """
    并行RNN层：多个RNN路径并行计算（仅单向，用于时间序列预测）
    """

    def __init__(
            self,
            input_dim: int,
            rnn_types: Union[str, List[str]] = 'gru',
            hidden_dims: Union[int, List[int]] = 256,
            num_layers_list: Union[int, List[int]] = 2,
            dropout_list: Union[float, List[float]] = 0.3,
            aggregation_method: str = 'concat',  # 'concat', 'sum', 'mean', 'max', 'attention'
            use_attention_aggregation: bool = False,  # 是否使用注意力聚合
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

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

        # 扩展列表长度
        def extend_list(lst, target_len):
            if len(lst) == 1:
                return lst * target_len
            assert len(lst) == target_len
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
                bidirectional=False,  # 时间序列预测必须是单向的！
                dropout=dropout_list[i] if num_layers_list[i] > 1 else 0.0
            )
            self.rnn_paths.append(rnn)
            self.output_dims.append(hidden_dims[i])  # 单向RNN，输出维度等于hidden_dim

        # 注意力聚合（如果需要）
        if use_attention_aggregation:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[0], 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # 聚合层
        if aggregation_method == 'concat':
            self.aggregation_dim = sum(self.output_dims)
        elif aggregation_method in ['sum', 'mean', 'max', 'attention']:
            self.aggregation_dim = self.output_dims[0]  # 所有路径输出维度必须相同
            for dim in self.output_dims[1:]:
                assert dim == self.aggregation_dim, \
                    f"For {aggregation_method} aggregation, all paths must have same output dim"
        else:
            raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

        self.to(device)

    def forward(self, x):
        # 并行计算所有路径
        outputs = []
        hidden_states = []

        for rnn_path in self.rnn_paths:
            rnn_out, hidden_state = rnn_path(x)
            outputs.append(rnn_out)
            hidden_states.append(hidden_state)

        # 聚合
        if self.aggregation_method == 'concat':
            aggregated = torch.cat(outputs, dim=-1)
        elif self.aggregation_method == 'sum':
            aggregated = sum(outputs)
        elif self.aggregation_method == 'mean':
            aggregated = torch.stack(outputs, dim=0).mean(dim=0)
        elif self.aggregation_method == 'max':
            aggregated = torch.stack(outputs, dim=0).max(dim=0)[0]
        elif self.aggregation_method == 'attention' and self.use_attention_aggregation:
            # 注意力加权聚合
            weights = []
            for i, output in enumerate(outputs):
                # 计算每个路径的重要性分数
                # 使用最后一个时间步的特征
                last_output = output[:, -1, :]  # (B, D)
                weight = self.attention(last_output)  # (B, 1)
                weights.append(weight)

            weights = torch.stack(weights, dim=1)  # (B, num_paths, 1)
            weights = F.softmax(weights, dim=1)

            # 加权聚合
            stacked = torch.stack(outputs, dim=0)  # (num_paths, B, L, D)
            weights_expanded = weights.unsqueeze(2).unsqueeze(3)  # (B, num_paths, 1, 1)
            aggregated = (stacked.permute(1, 0, 2, 3) * weights_expanded).sum(dim=1)
        else:
            aggregated = outputs[0]

        return aggregated, hidden_states


class AdvancedDNNCausalCNNRNNParallel(nn.Module):
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
            input_dim: int,
            input_seq_len: int = 600,

            # DNN1配置（支持多路径）
            dnn1_hidden_dims_list: List[List[int]] = None,
            dnn1_activations: Union[str, List[str]] = 'relu',
            dnn1_dropouts: Union[float, List[float]] = 0.1,
            dnn1_use_layer_norms: Union[bool, List[bool]] = True,
            dnn1_aggregation_method: str = 'concat',

            # CNN配置（支持多路径）
            cnn_hidden_dims_list: List[List[int]] = None,
            cnn_kernel_sizes_list: List[List[int]] = None,
            cnn_dilations_list: Optional[List[List[int]]] = None,
            cnn_strides_list: Optional[List[List[int]]] = None,
            cnn_use_skip_connections_list: Union[bool, List[bool]] = True,
            cnn_aggregation_method: str = 'concat',

            # DNN2配置（支持多路径）
            dnn2_hidden_dims_list: List[List[int]] = None,
            dnn2_activations: Union[str, List[str]] = 'relu',
            dnn2_dropouts: Union[float, List[float]] = 0.2,
            dnn2_use_layer_norms: Union[bool, List[bool]] = True,
            dnn2_aggregation_method: str = 'concat',

            # RNN配置（支持多路径，仅单向）
            rnn_types: Union[str, List[str]] = 'gru',
            rnn_hidden_dims: Union[int, List[int]] = 256,
            rnn_num_layers_list: Union[int, List[int]] = 2,
            rnn_dropout_list: Union[float, List[float]] = 0.3,
            rnn_aggregation_method: str = 'concat',
            rnn_use_attention_aggregation: bool = False,

            # 输出配置
            output_dim: Optional[int] = None,
            output_type: str = 'regression',  # 时间序列预测通常是回归
            num_classes: Optional[int] = None,
            return_sequences: bool = False,

            # 残差连接配置（加速收敛）
            use_residual_connections: bool = False,
            residual_strength: float = 0.1,  # 残差连接强度

            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_seq_len = input_seq_len
        self.output_type = output_type
        self.return_sequences = return_sequences
        self.num_classes = num_classes
        self.use_residual_connections = use_residual_connections
        self.residual_strength = residual_strength
        self.device = device

        if output_type == 'classification' and num_classes is None:
            raise ValueError("num_classes must be provided for classification.")

        # 设置默认值
        if dnn1_hidden_dims_list is None:
            dnn1_hidden_dims_list = [[32, 64, 128]]

        if cnn_hidden_dims_list is None:
            cnn_hidden_dims_list = [[128, 256, 512]]
        if cnn_kernel_sizes_list is None:
            cnn_kernel_sizes_list = [[3, 5, 7]]

        if dnn2_hidden_dims_list is None:
            dnn2_hidden_dims_list = [[256, 128]]

        current_seq_len = input_seq_len
        self.seq_len_history = [('input', current_seq_len)]

        # ===== DNN1 (并行多路径) =====
        self.dnn1 = ParallelDNNLayer(
            input_dim=input_dim,
            hidden_dims_list=dnn1_hidden_dims_list,
            activations=dnn1_activations,
            dropouts=dnn1_dropouts,
            use_layer_norms=dnn1_use_layer_norms,
            aggregation_method=dnn1_aggregation_method,
            device=device
        )
        dnn1_out_dim = self.dnn1.aggregation_dim

        # ===== CNN (并行多路径) =====
        self.cnn = ParallelCausalCNNBlock(
            input_dim=dnn1_out_dim,
            hidden_dims_list=cnn_hidden_dims_list,
            kernel_sizes_list=cnn_kernel_sizes_list,
            dilations_list=cnn_dilations_list,
            strides_list=cnn_strides_list,
            use_skip_connections_list=cnn_use_skip_connections_list,
            aggregation_method=cnn_aggregation_method,
            device=device
        )

        # 计算CNN后的序列长度
        current_seq_len = self.cnn.calculate_output_length(current_seq_len)
        self.seq_len_history.append(('cnn', current_seq_len))
        self.cnn_output_len = current_seq_len

        # ===== DNN2 (并行多路径) =====
        self.dnn2 = ParallelDNNLayer(
            input_dim=self.cnn.aggregation_dim,
            hidden_dims_list=dnn2_hidden_dims_list,
            activations=dnn2_activations,
            dropouts=dnn2_dropouts,
            use_layer_norms=dnn2_use_layer_norms,
            aggregation_method=dnn2_aggregation_method,
            device=device
        )
        dnn2_out_dim = self.dnn2.aggregation_dim

        # ===== RNN (并行多路径，仅单向) =====
        self.rnn = ParallelRNNLayer(
            input_dim=dnn2_out_dim,
            rnn_types=rnn_types,
            hidden_dims=rnn_hidden_dims,
            num_layers_list=rnn_num_layers_list,
            dropout_list=rnn_dropout_list,
            aggregation_method=rnn_aggregation_method,
            use_attention_aggregation=rnn_use_attention_aggregation,
            device=device
        )
        rnn_out_dim = self.rnn.aggregation_dim

        # ===== 输出层 =====
        if output_type == 'classification':
            final_out_dim = num_classes
        elif output_dim is not None:
            final_out_dim = output_dim
        else:
            final_out_dim = 1  # 时间序列预测通常是单值回归

        self.final_out_dim = final_out_dim

        if return_sequences:
            # 序列输出：每个时间步都预测
            self.output_layer = nn.Sequential(
                nn.Linear(rnn_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, final_out_dim)
            )
        else:
            # 单值输出：只预测最后一个时间步
            layers = []
            if rnn_out_dim != final_out_dim:
                layers.extend([
                    nn.Linear(rnn_out_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, final_out_dim)
                ])
            else:
                layers.append(nn.Linear(rnn_out_dim, final_out_dim))

            if output_type == 'classification':
                layers.append(nn.Softmax(dim=-1) if num_classes > 2 else nn.Sigmoid())
            self.output_layer = nn.Sequential(*layers)

        # 模块间残差连接（加速收敛）
        if use_residual_connections:
            # DNN1到CNN的残差连接投影
            self.dnn1_to_cnn_residual = self._create_residual_projection(
                input_dim, dnn1_out_dim, "dnn1_to_cnn"
            )

            # CNN到DNN2的残差连接投影
            self.cnn_to_dnn2_residual = self._create_residual_projection(
                dnn1_out_dim, dnn2_out_dim, "cnn_to_dnn2"
            )

            # DNN2到RNN的残差连接投影
            self.dnn2_to_rnn_residual = self._create_residual_projection(
                dnn2_out_dim, rnn_out_dim, "dnn2_to_rnn"
            )

        self.to(device)

    def _create_residual_projection(self, in_dim, out_dim, name):
        """创建残差连接投影层"""
        if in_dim == out_dim:
            return nn.Identity()
        else:
            # 使用带激活函数的投影，更稳定
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def forward(self, x, return_features=False):
        features = {}

        # DNN1
        dnn1_out = self.dnn1(x)
        if return_features: features['dnn1'] = dnn1_out

        # CNN (带可能的残差连接)
        cnn_input = dnn1_out
        if self.use_residual_connections:
            residual = self.dnn1_to_cnn_residual(x)
            cnn_input = cnn_input + self.residual_strength * residual

        cnn_out = self.cnn(cnn_input)
        if return_features: features['cnn'] = cnn_out

        # DNN2 (带可能的残差连接)
        dnn2_input = cnn_out
        if self.use_residual_connections:
            residual = self.cnn_to_dnn2_residual(dnn1_out)
            dnn2_input = dnn2_input + self.residual_strength * residual

        dnn2_out = self.dnn2(dnn2_input)
        if return_features: features['dnn2'] = dnn2_out

        # RNN (带可能的残差连接)
        rnn_input = dnn2_out
        if self.use_residual_connections:
            residual = self.dnn2_to_rnn_residual(cnn_out)
            rnn_input = rnn_input + self.residual_strength * residual

        rnn_out, rnn_hidden = self.rnn(rnn_input)
        if return_features:
            features['rnn_out'] = rnn_out
            features['rnn_hidden'] = rnn_hidden

        # 输出层
        if self.return_sequences:
            out = self.output_layer(rnn_out)
        else:
            out = self.output_layer(rnn_out[:, -1, :])

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


# ==================== Linear DNN ====================

class LinearDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu',
                 dropout=0.1, use_layer_norm=True):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
            nn.init.zeros_(layers[-1].bias)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            dr = dropout if isinstance(dropout, float) else dropout[0]
            if dr > 0:
                layers.append(nn.Dropout(dr))
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==================== Example ====================

if __name__ == "__main__":
    model = AdvancedDNNCausalCNNRNN(
        input_dim=20,
        input_seq_len=600,
        dnn1_hidden_dims=[32, 64,128],
        cnn_hidden_dims=[128,128],
        cnn_kernel_sizes=[3, 5, 14],
        cnn_dilations=[1, 2],
        cnn_strides=[1, 1],  # 第二层 stride=2，会下采样
        dnn2_hidden_dims=[128, 64],
        rnn_hidden_dim=128,
        output_type='classification',
        num_classes=3,
        return_sequences=False
    )

    print("Seq len info:", model.get_seq_len_info())
    # 示例输出:
    #   input=600 → after cnn: (600-1)//1+1 = 600 → (600-1)//2+1 = 300

    x = torch.randn(4, 600, 20)
    out = model(x)
    print("Input:", x.shape, "Output:", out.shape)

import yaml
from typing import Dict, Any
from dataclasses import dataclass, field, asdict
import torch
from pathlib import Path


@dataclass
class DNNParallelConfig:
    """并行DNN配置"""
    hidden_dims_list: List[List[int]] = field(default_factory=lambda: [[32, 64, 128]])
    activations: Union[str, List[str]] = "relu"
    dropouts: Union[float, List[float]] = 0.1
    use_layer_norms: Union[bool, List[bool]] = True
    aggregation_method: str = "concat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNNParallelConfig":
        return cls(**data)


@dataclass
class CNNParallelConfig:
    """并行CNN配置"""
    hidden_dims_list: List[List[int]] = field(default_factory=lambda: [[128, 256, 512]])
    kernel_sizes_list: List[List[int]] = field(default_factory=lambda: [[3, 5, 7]])
    dilations_list: Optional[List[List[int]]] = None
    strides_list: Optional[List[List[int]]] = None
    use_skip_connections_list: Union[bool, List[bool]] = True
    aggregation_method: str = "concat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CNNParallelConfig":
        return cls(**data)


@dataclass
class RNNParallelConfig:
    """并行RNN配置（时间序列预测专用）"""
    rnn_types: Union[str, List[str]] = "gru"
    hidden_dims: Union[int, List[int]] = 256
    num_layers_list: Union[int, List[int]] = 2
    dropout_list: Union[float, List[float]] = 0.3
    aggregation_method: str = "concat"
    use_attention_aggregation: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RNNParallelConfig":
        return cls(**data)


@dataclass
class ResidualConfig:
    """残差连接配置"""
    enabled: bool = False
    strength: float = 0.1
    skip_connections: List[str] = field(default_factory=lambda: ["dnn1_to_cnn", "cnn_to_dnn2"])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidualConfig":
        return cls(**data)


@dataclass
class OutputConfig:
    """输出层配置"""
    output_type: str = "regression"  # regression, classification
    output_dim: Optional[int] = None
    num_classes: Optional[int] = None
    return_sequences: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(**data)


@dataclass
class TimeSeriesModelConfig:
    """时间序列预测模型完整配置"""
    # === 基础配置 ===
    model_name: str = "TimeSeriesDNNCausalCNNRNNParallel"
    description: str = "时间序列预测模型"
    version: str = "1.0.0"

    # === 输入配置 ===
    input_dim: int = 20
    input_seq_len: int = 600

    # === 模块配置 ===
    dnn1: DNNParallelConfig = field(default_factory=DNNParallelConfig)
    cnn: CNNParallelConfig = field(default_factory=CNNParallelConfig)
    dnn2: DNNParallelConfig = field(default_factory=lambda: DNNParallelConfig(
        hidden_dims_list=[[256, 128]],
        dropouts=0.2
    ))
    rnn: RNNParallelConfig = field(default_factory=RNNParallelConfig)

    # === 输出配置 ===
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        output_type="regression",
        output_dim=1
    ))

    # === 残差连接配置 ===
    residual: ResidualConfig = field(default_factory=ResidualConfig)

    # === 训练配置 ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # === 元数据 ===
    created_date: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesModelConfig":
        """从字典创建配置对象"""
        # 处理嵌套配置
        config_dict = data.copy()

        for key, config_class in [
            ("dnn1", DNNParallelConfig),
            ("cnn", CNNParallelConfig),
            ("dnn2", DNNParallelConfig),
            ("rnn", RNNParallelConfig),
            ("residual", ResidualConfig),
            ("output", OutputConfig)
        ]:
            if key in config_dict and isinstance(config_dict[key], dict):
                config_dict[key] = config_class.from_dict(config_dict[key])

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = asdict(value)
            else:
                result[key] = value
        return result

    def save_yaml(self, filepath: Union[str, Path]):
        """保存为YAML文件"""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, filepath: Union[str, Path]) -> "TimeSeriesModelConfig":
        """从YAML文件加载"""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class ModelConfigManager:
    """模型配置管理器"""

    @staticmethod
    def create_model_from_yaml(
            config_path: Union[str, Path],
            model_class
    ) -> AdvancedDNNCausalCNNRNNParallel:
        """
        从YAML配置文件创建模型

        Args:
            config_path: YAML配置文件路径
            model_class: 模型类

        Returns:
            创建的模型实例
        """
        # 加载配置
        config = TimeSeriesModelConfig.load_yaml(config_path)

        # 转换为模型参数
        model_kwargs = ModelConfigManager._config_to_kwargs(config)

        # 创建模型
        return model_class(**model_kwargs)

    @staticmethod
    def create_default_config() -> TimeSeriesModelConfig:
        """创建默认配置"""
        return TimeSeriesModelConfig()

    @staticmethod
    def _config_to_kwargs(config: TimeSeriesModelConfig) -> Dict[str, Any]:
        """将配置对象转换为模型参数字典"""
        kwargs = {
            "input_dim": config.input_dim,
            "input_seq_len": config.input_seq_len,

            # DNN1
            "dnn1_hidden_dims_list": config.dnn1.hidden_dims_list,
            "dnn1_activations": config.dnn1.activations,
            "dnn1_dropouts": config.dnn1.dropouts,
            "dnn1_use_layer_norms": config.dnn1.use_layer_norms,
            "dnn1_aggregation_method": config.dnn1.aggregation_method,

            # CNN
            "cnn_hidden_dims_list": config.cnn.hidden_dims_list,
            "cnn_kernel_sizes_list": config.cnn.kernel_sizes_list,
            "cnn_dilations_list": config.cnn.dilations_list,
            "cnn_strides_list": config.cnn.strides_list,
            "cnn_use_skip_connections_list": config.cnn.use_skip_connections_list,
            "cnn_aggregation_method": config.cnn.aggregation_method,

            # DNN2
            "dnn2_hidden_dims_list": config.dnn2.hidden_dims_list,
            "dnn2_activations": config.dnn2.activations,
            "dnn2_dropouts": config.dnn2.dropouts,
            "dnn2_use_layer_norms": config.dnn2.use_layer_norms,
            "dnn2_aggregation_method": config.dnn2.aggregation_method,

            # RNN
            "rnn_types": config.rnn.rnn_types,
            "rnn_hidden_dims": config.rnn.hidden_dims,
            "rnn_num_layers_list": config.rnn.num_layers_list,
            "rnn_dropout_list": config.rnn.dropout_list,
            "rnn_aggregation_method": config.rnn.aggregation_method,
            "rnn_use_attention_aggregation": config.rnn.use_attention_aggregation,

            # Output
            "output_type": config.output.output_type,
            "output_dim": config.output.output_dim,
            "num_classes": config.output.num_classes,
            "return_sequences": config.output.return_sequences,

            # Residual
            "use_residual_connections": config.residual.enabled,
            "residual_strength": config.residual.strength,

            # Device
            "device": config.device,
        }

        # 移除为None的参数
        return {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def generate_example_config(output_path: Union[str, Path] = "model_config_example.yaml"):
        """生成详细的示例配置文件"""
        from datetime import datetime

        example_config = TimeSeriesModelConfig(
            model_name="ExampleTimeSeriesModel",
            description="时间序列预测示例配置 - 包含所有配置项和详细说明",
            version="1.0.0",
            input_dim=30,
            input_seq_len=500,

            dnn1=DNNParallelConfig(
                hidden_dims_list=[[32, 64, 128], [64, 128, 256]],
                activations="relu",
                dropouts=[0.1, 0.15],
                use_layer_norms=[True, True],
                aggregation_method="concat"
            ),

            cnn=CNNParallelConfig(
                hidden_dims_list=[[128, 256, 512], [256, 512, 1024]],
                kernel_sizes_list=[[3, 5, 7], [5, 7, 9]],
                dilations_list=[[1, 1, 1], [1, 2, 4]],
                strides_list=[[1, 1, 1], [1, 1, 1]],
                use_skip_connections_list=[True, True],
                aggregation_method="concat"
            ),

            dnn2=DNNParallelConfig(
                hidden_dims_list=[[512, 256], [256, 128]],
                activations="relu",
                dropouts=0.2,
                use_layer_norms=True,
                aggregation_method="concat"
            ),

            rnn=RNNParallelConfig(
                rnn_types=["gru", "lstm"],
                hidden_dims=[256, 256],
                num_layers_list=[2, 2],
                dropout_list=[0.3, 0.3],
                aggregation_method="attention",
                use_attention_aggregation=True
            ),

            output=OutputConfig(
                output_type="regression",
                output_dim=1,
                return_sequences=False
            ),

            residual=ResidualConfig(
                enabled=True,
                strength=0.15,
                skip_connections=["dnn1_to_cnn", "cnn_to_dnn2", "dnn2_to_rnn"]
            ),

            device="cuda" if torch.cuda.is_available() else "cpu",
            created_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            author="TimeSeriesAI",
            tags=["forecasting", "multi-scale", "parallel"]
        )

        # 添加注释
        config_dict = example_config.to_dict()
        output_path = Path(output_path)

        # 生成带注释的YAML
        yaml_str = ModelConfigManager._generate_commented_yaml(config_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_str)

        print(f"示例配置文件已生成: {output_path}")
        return example_config

    @staticmethod
    def _generate_commented_yaml(config_dict: Dict[str, Any], indent: str = "") -> str:
        """生成带注释的YAML字符串"""
        comments = {
            "model_name": "模型名称，用于标识",
            "description": "模型描述，说明模型用途和特点",
            "version": "配置版本",
            "input_dim": "输入特征维度，每个时间步的特征数量",
            "input_seq_len": "输入序列长度，时间步数量",

            "dnn1": "第一个DNN模块配置（特征提取）",
            "dnn1.hidden_dims_list": "每个DNN路径的隐藏层维度列表，例如[[32,64,128],[64,128,256]]表示两个路径",
            "dnn1.activations": "激活函数，可以是单个字符串或列表",
            "dnn1.dropouts": "dropout率，防止过拟合",
            "dnn1.use_layer_norms": "是否使用LayerNorm",
            "dnn1.aggregation_method": "多路径聚合方法：concat(拼接), sum(求和), mean(平均), max(最大值)",

            "cnn": "CNN模块配置（时序特征提取）",
            "cnn.hidden_dims_list": "每个CNN路径的隐藏层维度",
            "cnn.kernel_sizes_list": "每个CNN路径的卷积核大小",
            "cnn.dilations_list": "扩张率，用于捕捉长距离依赖",
            "cnn.strides_list": "步幅，用于下采样",
            "cnn.use_skip_connections_list": "是否使用残差连接（CNN内部）",
            "cnn.aggregation_method": "多路径聚合方法",

            "dnn2": "第二个DNN模块配置（特征变换）",

            "rnn": "RNN模块配置（时序建模）",
            "rnn.rnn_types": "RNN类型：gru, lstm, rnn",
            "rnn.hidden_dims": "隐藏层维度",
            "rnn.num_layers_list": "层数",
            "rnn.dropout_list": "RNN dropout",
            "rnn.aggregation_method": "聚合方法，attention表示使用注意力机制",
            "rnn.use_attention_aggregation": "是否使用注意力聚合",

            "output": "输出层配置",
            "output.output_type": "输出类型：regression(回归), classification(分类)",
            "output.output_dim": "输出维度，回归任务通常为1",
            "output.num_classes": "分类任务的类别数",
            "output.return_sequences": "是否返回整个序列（False时只返回最后一个时间步）",

            "residual": "模块间残差连接配置（加速收敛）",
            "residual.enabled": "是否启用残差连接",
            "residual.strength": "残差连接强度，0-1之间",
            "residual.skip_connections": "残差连接路径",

            "device": "运行设备：cuda, cpu",
            "created_date": "创建日期",
            "author": "作者",
            "tags": "标签"
        }

        lines = []
        for key, value in config_dict.items():
            # 添加注释
            if key in comments:
                lines.append(f"# {comments[key]}")

            if isinstance(value, dict):
                lines.append(f"{key}:")
                sub_lines = ModelConfigManager._generate_commented_yaml(value, indent + "  ")
                lines.append(sub_lines)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{key}:")
                for i, item in enumerate(value):
                    lines.append(f"{indent}  -")
                    sub_lines = ModelConfigManager._generate_commented_yaml(item, indent + "    ")
                    lines.append(sub_lines)
            else:
                lines.append(f"{key}: {yaml.dump({key: value}, default_flow_style=None).split(': ')[1].strip()}")

        return "\n".join(lines)


# ==================== 使用示例 ====================

def create_model_from_yaml_demo():
    """演示从YAML创建模型的完整流程"""

    # 1. 生成示例配置文件
    print("1. 生成示例配置文件...")
    ModelConfigManager.generate_example_config("model_config_example.yaml")

    # 2. 从YAML文件创建模型
    print("\n2. 从YAML文件创建模型...")
    try:
        model = ModelConfigManager.create_model_from_yaml(
            "model_config_example.yaml",
            AdvancedDNNCausalCNNRNNParallel
        )
        print(f"✓ 模型创建成功！")
        print(f"  模型名称: {model.__class__.__name__}")
        print(f"  输入维度: {model.input_dim}")
        print(f"  序列长度: {model.input_seq_len}")
        print(f"  使用残差连接: {model.use_residual_connections}")

        # 测试模型
        x = torch.randn(4, model.input_seq_len, model.input_dim)
        output = model(x)
        print(f"  测试输入: {x.shape} → 输出: {output.shape}")

    except FileNotFoundError:
        print("✗ 请先生成示例配置文件")

    # 3. 创建自定义配置
    print("\n3. 创建自定义配置...")
    custom_config = TimeSeriesModelConfig(
        model_name="CustomForecastModel",
        description="自定义时间序列预测模型",
        input_dim=40,
        input_seq_len=300,

        dnn1=DNNParallelConfig(
            hidden_dims_list=[[64, 128]],
            aggregation_method="concat"
        ),

        cnn=CNNParallelConfig(
            hidden_dims_list=[[128, 256]],
            kernel_sizes_list=[[3, 5]],
            dilations_list=[[1, 2]]
        ),

        output=OutputConfig(
            output_type="regression",
            output_dim=1
        ),

        residual=ResidualConfig(
            enabled=True,
            strength=0.2
        )
    )

    # 保存自定义配置
    custom_config.save_yaml("custom_config.yaml")
    print(f"✓ 自定义配置已保存到 custom_config.yaml")

    # 4. 从自定义配置创建模型
    print("\n4. 从自定义配置创建模型...")
    custom_model = ModelConfigManager.create_model_from_yaml(
        "custom_config.yaml",
        AdvancedDNNCausalCNNRNNParallel
    )
    print(f"✓ 自定义模型创建成功！")


def get_time_series_preset_config(preset_name: str) -> Dict[str, Any]:
    """获取时间序列预测专用预设配置"""
    presets = {
        "multi_scale_time_series": {
            "description": "多尺度时间序列预测 - 捕捉不同时间尺度模式",
            "cnn": {
                "hidden_dims_list": [
                    [64, 128, 256],  # 短期模式（小核）
                    [64, 128, 256],  # 中期模式（中核）
                    [64, 128, 256]  # 长期模式（大核）
                ],
                "kernel_sizes_list": [
                    [3, 5, 7],  # 短期模式
                    [5, 7, 9],  # 中期模式
                    [7, 9, 11]  # 长期模式
                ],
                "dilations_list": [
                    [1, 1, 1],  # 无扩张
                    [1, 2, 4],  # 扩张卷积捕捉长期依赖
                    [1, 3, 6]  # 更大扩张
                ],
                "aggregation_method": "concat"
            },
            "rnn": {
                "rnn_types": ["gru", "lstm"],  # 混合RNN类型
                "hidden_dims": [128, 128],
                "use_attention_aggregation": True  # 使用注意力选择重要特征
            }
        },

        "fast_convergence": {
            "description": "快速收敛配置 - 使用残差连接加速训练",
            "residual": {
                "enabled": True,
                "strength": 0.2,
                "skip_connections": ["dnn1_to_cnn", "cnn_to_dnn2", "dnn2_to_rnn"]
            },
            "dnn1": {
                "hidden_dims_list": [[32, 64], [64, 128]],  # 两个路径
                "aggregation_method": "sum"  # 使用sum以便残差连接
            },
            "cnn": {
                "hidden_dims_list": [[128, 256], [256, 512]],
                "kernel_sizes_list": [[3, 5], [5, 7]],
                "aggregation_method": "sum"
            }
        },

        "robust_forecasting": {
            "description": "稳健预测配置 - 防止过拟合，提升泛化",
            "dnn1": {
                "dropouts": 0.2,  # 提高dropout
                "use_layer_norms": True
            },
            "cnn": {
                "use_skip_connections_list": True
            },
            "dnn2": {
                "dropouts": 0.3
            },
            "rnn": {
                "dropout_list": 0.4,  # RNN dropout
                "num_layers_list": 2
            }
        },

        "lightweight_forecast": {
            "description": "轻量级预测配置 - 适合实时预测",
            "dnn1": {
                "hidden_dims_list": [[16, 32, 64]],
                "use_layer_norms": False
            },
            "cnn": {
                "hidden_dims_list": [[64, 128]],
                "kernel_sizes_list": [[3, 5]],
                "strides_list": [[1, 2]]  # 使用步幅减少计算
            },
            "dnn2": {
                "hidden_dims_list": [[64, 32]]
            },
            "rnn": {
                "hidden_dims": 64,
                "num_layers_list": 1
            }
        },

        "multi_horizon": {
            "description": "多步预测配置 - 同时预测多个时间步",
            "output": {
                "return_sequences": True,  # 输出整个序列
                "output_dim": 10  # 预测未来10个时间步
            },
            "rnn": {
                "hidden_dims": 512,  # 更大容量处理多步预测
                "num_layers_list": 3
            }
        },

        "volatility_aware": {
            "description": "波动率感知配置 - 适合波动大的时间序列",
            "cnn": {
                "hidden_dims_list": [[128, 256], [128, 256]],
                "kernel_sizes_list": [[3, 5], [7, 9]],
                "dilations_list": [[1, 1], [1, 2]]  # 捕捉不同频率波动
            },
            "rnn": {
                "rnn_types": ["lstm"],  # LSTM更适合长序列
                "hidden_dims": 256,
                "num_layers_list": 2
            },
            "residual": {
                "enabled": True,
                "strength": 0.15
            }
        }
    }