# 1、使用AdvancedDNNCausalCNNRNNParallel处理低频数据 -> 低频预测结果，
# 2、使用AdvancedDNNCausalCNNRNNParallel 处理中频数据，处理时使用是否使用FiLM，并将低频预测结果作为lf_context -> 中频预测解结果
# 3、concat 低频、中频预测结果，输入多层DNN网络进行最后预测，预测结果是多标签二分类


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import yaml

from data_alchemy.AutoFactorAndRnnModelWithFiLM import AdvancedDNNCausalCNNRNNParallelWithFiLM
from data_alchemy.utils import get_device


class MultiFreqMultiLabelClassifier(nn.Module):
    """
    多频段时间序列多标签二分类模型
    1. 处理低频数据 -> 低频特征
    2. 处理中频数据，使用FiLM调制（以低频特征作为上下文）-> 中频特征
    3. 合并特征 -> 多个独立的二分类器
    """

    def __init__(
            self,
            # 低频配置
            lf_input_dim: int,
            lf_seq_len: int,

            # 中频配置
            mf_input_dim: int,
            mf_seq_len: int,
            lf_feature_dim: int,  # 低频特征提取维度
            mf_feature_dim: int,  # 中频特征提取维度

            # 标签配置
            num_labels:int,  # 二分类标签数量

            # 低频模型配置
            lf_dnn1_hidden_dims_list: List[List[int]],
            lf_cnn_hidden_dims_list: List[List[int]] ,
            lf_cnn_kernel_sizes_list: List[List[int]],
            lf_rnn_hidden_dims: int,

            # 中频模型配置
            mf_dnn1_hidden_dims_list: List[List[int]],
            mf_cnn_hidden_dims_list: List[List[int]],
            mf_cnn_kernel_sizes_list: List[List[int]],
            mf_rnn_hidden_dims: int,
            mf_use_residual: bool,

            # 共享特征提取器配置
            shared_hidden_dims: List[int],  # 共享特征提取层

            # 分类头配置
            per_label_hidden_dims: List[int],  # 每个标签的分类头

            # 其他配置
            dropout: float,
            use_batch_norm: bool,
            label_dropout: float,  # 标签间的dropout
            device: str = 'auto',
            init_seed: Optional[int] = None
    ):
        super().__init__()

        if device == 'auto':
            device = get_device()
        self.device = device

        self.num_labels = num_labels
        self.lf_feature_dim = lf_feature_dim
        self.mf_feature_dim = mf_feature_dim
        self.per_label_hidden_dims = per_label_hidden_dims

        # 设置随机种子
        if init_seed is not None:
            self._set_init_seed(init_seed)

        print("=" * 70)
        print("构建多频段多标签二分类模型")
        print(f"标签数量: {num_labels}")
        print("=" * 70)

        # ===== 1. 低频特征提取器 =====
        print("\n1. 低频特征提取器")
        print(f"   输入: ({lf_seq_len}, {lf_input_dim})")
        print(f"   特征输出: {lf_feature_dim}维")

        self.lf_extractor = AdvancedDNNCausalCNNRNNParallelWithFiLM(
            input_dim=lf_input_dim,
            input_seq_len=lf_seq_len,
            output_dim=lf_feature_dim,
            output_type='regression',
            return_sequences=False,

            # DNN1
            dnn1_hidden_dims_list=lf_dnn1_hidden_dims_list,
            dnn1_aggregation_method='concat',

            # CNN
            cnn_hidden_dims_list=lf_cnn_hidden_dims_list,
            cnn_kernel_sizes_list=lf_cnn_kernel_sizes_list,
            cnn_aggregation_method='concat',

            # RNN
            rnn_hidden_dims=lf_rnn_hidden_dims,
            rnn_aggregation_method='concat',

            use_film=False,
            device=device
        )

        # ===== 2. 中频特征提取器（使用FiLM） =====
        print("\n2. 中频特征提取器（使用FiLM调制）")
        print(f"   输入: ({mf_seq_len}, {mf_input_dim})")
        print(f"   特征输出: {mf_feature_dim}维")
        print(f"   FiLM上下文: 低频特征 ({lf_feature_dim}维)")

        self.mf_extractor = AdvancedDNNCausalCNNRNNParallelWithFiLM(
            input_dim=mf_input_dim,
            input_seq_len=mf_seq_len,
            output_dim=mf_feature_dim,
            output_type='regression',
            return_sequences=False,

            # DNN1
            dnn1_hidden_dims_list=mf_dnn1_hidden_dims_list,
            dnn1_aggregation_method='concat',

            # CNN
            cnn_hidden_dims_list=mf_cnn_hidden_dims_list,
            cnn_kernel_sizes_list=mf_cnn_kernel_sizes_list,
            cnn_aggregation_method='concat',

            # RNN
            rnn_hidden_dims=mf_rnn_hidden_dims,
            rnn_aggregation_method='concat',

            use_film=True,
            lf_context_dim=lf_feature_dim,
            use_residual_connections=mf_use_residual,
            device=device
        )

        # ===== 3. 共享特征提取器 =====
        print("\n3. 共享特征提取器")
        shared_input_dim = lf_feature_dim + mf_feature_dim
        print(f"   输入维度: {shared_input_dim}")
        print(f"   隐藏层: {shared_hidden_dims}")

        self.shared_layers = nn.ModuleList()
        prev_dim = shared_input_dim

        for i, hidden_dim in enumerate(shared_hidden_dims):
            # 线性层
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            self.shared_layers.append(linear)

            # 批归一化
            if use_batch_norm:
                self.shared_layers.append(nn.BatchNorm1d(hidden_dim))

            # 激活函数
            self.shared_layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                self.shared_layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared_output_dim = prev_dim

        # ===== 4. 多标签分类头 =====
        print("\n4. 多标签分类头")
        print(f"   每个标签独立分类头")
        print(f"   每个头隐藏层: {per_label_hidden_dims}")
        print(f"   输出: {num_labels}个独立的二分类概率")

        self.label_heads = nn.ModuleList()

        for label_idx in range(num_labels):
            # 每个标签的独立分类器
            label_layers = []
            label_prev_dim = self.shared_output_dim

            # 隐藏层
            for i, hidden_dim in enumerate(per_label_hidden_dims):
                linear = nn.Linear(label_prev_dim, hidden_dim)
                nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
                nn.init.zeros_(linear.bias)
                label_layers.append(linear)

                if use_batch_norm:
                    label_layers.append(nn.BatchNorm1d(hidden_dim))

                label_layers.append(nn.ReLU())

                # 标签间dropout（防止标签间过度依赖）
                if label_dropout > 0:
                    label_layers.append(nn.Dropout(label_dropout))

                label_prev_dim = hidden_dim

            # 输出层：二分类（1个神经元）
            output_layer = nn.Linear(label_prev_dim, 1)
            nn.init.xavier_normal_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            label_layers.append(output_layer)

            # Sigmoid激活（每个标签独立）
            label_layers.append(nn.Sigmoid())

            # 创建这个标签的分类器
            label_classifier = nn.Sequential(*label_layers)
            self.label_heads.append(label_classifier)

            if label_idx < 3:  # 只显示前3个标签的详细信息
                print(f"   标签{label_idx}: {label_prev_dim} -> {per_label_hidden_dims} -> 1")

        if num_labels > 3:
            print(f"   ... 还有{num_labels - 3}个标签")

        # 移动到设备
        self.to(device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n总参数量: {total_params:,}")
        print("=" * 70)

    def _set_init_seed(self, seed: int):
        """设置初始化种子"""
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"初始化种子: {seed}")

    def forward(
            self,
            lf_input: torch.Tensor,  # 低频输入 (B, L_lf, D_lf)
            mf_input: torch.Tensor,  # 中频输入 (B, L_mf, D_mf)
            return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播

        Args:
            lf_input: 低频输入序列 [batch_size, lf_seq_len, lf_input_dim]
            mf_input: 中频输入序列 [batch_size, mf_seq_len, mf_input_dim]
            return_features: 是否返回中间特征

        Returns:
            多标签概率 [batch_size, num_labels]
            特征字典（如果return_features=True）
        """
        batch_size = lf_input.shape[0]
        features_dict = {} if return_features else None

        # 1. 提取低频特征
        lf_features = self.lf_extractor(lf_input)  # (B, lf_feature_dim)
        if return_features:
            features_dict['lf_features'] = lf_features

        # 2. 提取中频特征（使用低频特征作为FiLM上下文）
        mf_features = self.mf_extractor(mf_input, lf_context=lf_features)  # (B, mf_feature_dim)
        if return_features:
            features_dict['mf_features'] = mf_features

        # 3. 特征拼接
        combined = torch.cat([lf_features, mf_features], dim=-1)  # (B, lf_dim + mf_dim)
        if return_features:
            features_dict['combined_features'] = combined

        # 4. 共享特征提取
        x = combined
        for layer in self.shared_layers:
            x = layer(x)
        shared_features = x  # (B, shared_output_dim)
        if return_features:
            features_dict['shared_features'] = shared_features

        # 5. 多标签预测（每个标签独立）
        all_predictions = []
        for label_idx in range(self.num_labels):
            # 通过对应的标签头
            label_pred = self.label_heads[label_idx](shared_features)  # (B, 1)
            all_predictions.append(label_pred)

        # 拼接所有标签的预测
        predictions = torch.cat(all_predictions, dim=1)  # (B, num_labels)

        if return_features:
            features_dict['predictions'] = predictions
            return predictions, features_dict

        return predictions

    def predict_proba(self, lf_input: torch.Tensor, mf_input: torch.Tensor) -> torch.Tensor:
        """预测所有标签的概率"""
        self.eval()
        with torch.no_grad():
            probabilities = self(lf_input, mf_input)
        return probabilities

    def predict(self, lf_input: torch.Tensor, mf_input: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """预测所有标签的类别（0或1）"""
        probabilities = self.predict_proba(lf_input, mf_input)
        predictions = (probabilities > threshold).float()
        return predictions

    def predict_single_label(self, lf_input: torch.Tensor, mf_input: torch.Tensor,
                             label_idx: int, threshold: float = 0.5) -> torch.Tensor:
        """预测单个标签"""
        self.eval()
        with torch.no_grad():
            # 只计算单个标签，避免计算所有标签
            lf_features = self.lf_extractor(lf_input)
            mf_features = self.mf_extractor(mf_input, lf_context=lf_features)
            combined = torch.cat([lf_features, mf_features], dim=-1)

            # 共享特征
            x = combined
            for layer in self.shared_layers:
                x = layer(x)

            # 单个标签预测
            probability = self.label_heads[label_idx](x)
            prediction = (probability > threshold).float()

        return probability, prediction

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        return {
            'device': str(self.device),
            'num_labels': self.num_labels,
            'low_freq_extractor': {
                'input_shape': f"(B, {self.lf_extractor.input_seq_len}, {self.lf_extractor.input_dim})",
                'output_dim': self.lf_feature_dim,
            },
            'mid_freq_extractor': {
                'input_shape': f"(B, {self.mf_extractor.input_seq_len}, {self.mf_extractor.input_dim})",
                'output_dim': self.mf_feature_dim,
                'use_film': True,
            },
            'shared_extractor': {
                'input_dim': self.lf_feature_dim + self.mf_feature_dim,
                'output_dim': self.shared_output_dim,
                'hidden_dims': [module.out_features for module in self.shared_layers
                                if isinstance(module, nn.Linear)]
            },
            'per_label_classifiers': {
                'input_dim': self.shared_output_dim,
                'hidden_dims': self.per_label_hidden_dims,
                'output': 'binary_probability',
                'independent': True
            },
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


# ==================== 简化创建函数 ====================

def create_multi_label_classifier(
        lf_dim: int,
        mf_dim: int,
        num_labels: int,
        lf_len: int = 100,
        mf_len: int = 200,
        model_size: str = 'medium',  # 'small', 'medium', 'large'
        device: str = 'auto'
) -> MultiFreqMultiLabelClassifier:
    """
    快速创建多标签分类模型

    Args:
        lf_dim: 低频特征维度
        mf_dim: 中频特征维度
        num_labels: 二分类标签数量
        lf_len: 低频序列长度
        mf_len: 中频序列长度
        model_size: 模型大小
        device: 设备

    Returns:
        MultiFreqMultiLabelClassifier实例
    """

    # 根据模型大小设置配置
    if model_size == 'small':
        # 特征提取维度
        lf_feature_dim = 32
        mf_feature_dim = 64

        # 低频配置
        lf_dnn1 = [[16, 32]]
        lf_cnn = [[64, 128]]
        lf_rnn = 64

        # 中频配置
        mf_dnn1 = [[32, 64]]
        mf_cnn = [[128, 256]]
        mf_rnn = 128

        # 共享层
        shared_dims = [256, 128]

        # 每个标签头
        per_label_dims = [64, 32]

    elif model_size == 'medium':
        # 特征提取维度
        lf_feature_dim = 64
        mf_feature_dim = 128

        # 低频配置
        lf_dnn1 = [[32, 64]]
        lf_cnn = [[128, 256]]
        lf_rnn = 128

        # 中频配置
        mf_dnn1 = [[64, 128]]
        mf_cnn = [[256, 512]]
        mf_rnn = 256

        # 共享层
        shared_dims = [512, 256]

        # 每个标签头
        per_label_dims = [128, 64]

    else:  # 'large'
        # 特征提取维度
        lf_feature_dim = 128
        mf_feature_dim = 256

        # 低频配置
        lf_dnn1 = [[64, 128, 256]]
        lf_cnn = [[256, 512, 1024]]
        lf_rnn = 256

        # 中频配置
        mf_dnn1 = [[128, 256, 512]]
        mf_cnn = [[512, 1024, 2048]]
        mf_rnn = 512

        # 共享层
        shared_dims = [1024, 512, 256]

        # 每个标签头
        per_label_dims = [256, 128, 64]

    print(f"创建{model_size}尺寸的多标签分类模型...")
    print(f"低频: ({lf_len}, {lf_dim}) -> {lf_feature_dim}维特征")
    print(f"中频: ({mf_len}, {mf_dim}) -> {mf_feature_dim}维特征")
    print(f"标签数量: {num_labels}")
    print(f"每个标签独立分类器")

    model = MultiFreqMultiLabelClassifier(
        # 低频配置
        lf_input_dim=lf_dim,
        lf_seq_len=lf_len,
        lf_feature_dim=lf_feature_dim,

        # 中频配置
        mf_input_dim=mf_dim,
        mf_seq_len=mf_len,
        mf_feature_dim=mf_feature_dim,

        # 标签数量
        num_labels=num_labels,

        # 低频模型
        lf_dnn1_hidden_dims_list=lf_dnn1,
        lf_cnn_hidden_dims_list=lf_cnn,
        lf_cnn_kernel_sizes_list=[[3, 5, 7][:len(lf_cnn[0])]],
        lf_rnn_hidden_dims=lf_rnn,

        # 中频模型
        mf_dnn1_hidden_dims_list=mf_dnn1,
        mf_cnn_hidden_dims_list=mf_cnn,
        mf_cnn_kernel_sizes_list=[[5, 7, 9][:len(mf_cnn[0])]],
        mf_rnn_hidden_dims=mf_rnn,
        mf_use_residual=True,

        # 共享层
        shared_hidden_dims=shared_dims,

        # 每个标签头
        per_label_hidden_dims=per_label_dims,

        # 其他配置
        dropout=0.3,
        use_batch_norm=True,
        label_dropout=0.2,

        device=device
    )

    return model


# ==================== 训练工具类 ====================

class MultiLabelTrainer:
    """多标签分类模型训练工具"""

    def __init__(self, model: MultiFreqMultiLabelClassifier):
        self.model = model
        self.device = model.device

        # 损失函数：多标签二分类交叉熵
        self.criterion = nn.BCELoss(reduction='mean')

        # 优化器
        self.optimizer = None

    def setup_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        """设置优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_step(self, lf_batch: torch.Tensor, mf_batch: torch.Tensor,
                   labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """单步训练"""
        self.model.train()

        # 移到设备
        lf_batch = lf_batch.to(self.device)
        mf_batch = mf_batch.to(self.device)
        labels = labels.to(self.device).float()

        # 前向传播
        predictions = self.model(lf_batch, mf_batch)  # (B, num_labels)

        # 计算损失
        loss = self.criterion(predictions, labels)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 优化
        self.optimizer.step()

        return loss.item(), predictions

    def evaluate(self, lf_data: torch.Tensor, mf_data: torch.Tensor,
                 labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """评估模型"""
        self.model.eval()

        with torch.no_grad():
            lf_data = lf_data.to(self.device)
            mf_data = mf_data.to(self.device)
            labels = labels.to(self.device).float()

            # 预测
            probabilities = self.model(lf_data, mf_data)

            # 计算损失
            loss = self.criterion(probabilities, labels)

            # 转换为预测类别
            predictions = (probabilities > threshold).float()

            # 计算每个标签的准确率
            per_label_accuracy = []
            for i in range(self.model.num_labels):
                label_acc = (predictions[:, i] == labels[:, i]).float().mean().item()
                per_label_accuracy.append(label_acc)

            # 总体准确率（所有标签都正确）
            all_correct = (predictions == labels).all(dim=1).float().mean().item()

            # 平均准确率
            mean_accuracy = sum(per_label_accuracy) / len(per_label_accuracy)

            # 计算F1分数
            from sklearn.metrics import f1_score
            f1_micro = f1_score(labels.cpu().numpy(),
                                predictions.cpu().numpy(),
                                average='micro', zero_division=0)
            f1_macro = f1_score(labels.cpu().numpy(),
                                predictions.cpu().numpy(),
                                average='macro', zero_division=0)

            return {
                'loss': loss.item(),
                'mean_accuracy': mean_accuracy,
                'all_correct_accuracy': all_correct,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'per_label_accuracy': per_label_accuracy,
                'probabilities': probabilities.cpu(),
                'predictions': predictions.cpu()
            }


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    import torch

    print("多频段多标签二分类模型示例")
    print("=" * 60)

    # 1. 创建模型（假设有5个二分类标签）
    model = create_multi_label_classifier(
        lf_dim=10,  # 低频10个特征
        mf_dim=20,  # 中频20个特征
        num_labels=5,  # 5个二分类标签
        lf_len=100,  # 低频序列长度100
        mf_len=200,  # 中频序列长度200
        model_size='medium',
        device='cpu'
    )

    # 2. 模拟数据
    batch_size = 8
    lf_input = torch.randn(batch_size, 100, 10)
    mf_input = torch.randn(batch_size, 200, 20)
    labels = torch.randint(0, 2, (batch_size, 5)).float()  # 多标签：5个二分类

    print(f"\n输入数据形状:")
    print(f"  低频: {lf_input.shape}")
    print(f"  中频: {mf_input.shape}")
    print(f"  标签: {labels.shape} (多标签)")
    print(f"  标签示例: {labels[0].tolist()}")

    # 3. 前向传播测试
    model.eval()
    with torch.no_grad():
        # 获取预测概率
        probabilities = model(lf_input, mf_input)
        print(f"\n预测概率形状: {probabilities.shape}")
        print(f"概率范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

        # 获取所有特征
        probs, features = model(lf_input, mf_input, return_features=True)
        print(f"\n特征提取:")
        for name, tensor in features.items():
            print(f"  {name}: {tensor.shape}")

        # 预测类别
        predictions = model.predict(lf_input, mf_input, threshold=0.5)
        print(f"\n预测结果:")
        print(f"  样本0预测: {predictions[0].tolist()}")
        print(f"  样本0真实: {labels[0].tolist()}")

        # 预测单个标签
        prob_label0, pred_label0 = model.predict_single_label(lf_input, mf_input, label_idx=0)
        print(f"\n单个标签预测（标签0）:")
        print(f"  概率: {prob_label0[0].item():.3f}")
        print(f"  预测: {pred_label0[0].item()}")

    # 4. 创建训练器
    trainer = MultiLabelTrainer(model)
    trainer.setup_optimizer(lr=1e-3)

    # 5. 单步训练演示
    print("\n训练演示:")
    loss, preds = trainer.train_step(lf_input, mf_input, labels)
    print(f"  训练损失: {loss:.4f}")

    # 6. 评估演示
    print("\n评估演示:")
    metrics = trainer.evaluate(lf_input, mf_input, labels)
    print(f"  评估损失: {metrics['loss']:.4f}")
    print(f"  平均准确率: {metrics['mean_accuracy']:.3f}")
    print(f"  全部正确准确率: {metrics['all_correct_accuracy']:.3f}")
    print(f"  F1-micro: {metrics['f1_micro']:.3f}")
    print(f"  F1-macro: {metrics['f1_macro']:.3f}")

    # 7. 模型信息
    info = model.get_model_info()
    print(f"\n模型总参数量: {info['total_parameters']:,}")

    return model, trainer

