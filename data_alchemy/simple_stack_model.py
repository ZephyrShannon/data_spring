# 1、使用AdvancedDNNCausalCNNRNNParallel处理低频数据 -> 低频预测结果，
# 2、使用AdvancedDNNCausalCNNRNNParallel 处理中频数据，处理时使用是否使用FiLM，并将低频预测结果作为lf_context -> 中频预测解结果
# 3、concat 低频、中频预测结果，输入多层DNN网络进行最后预测，预测结果是多标签二分类


import torch
import torch.nn as nn
# import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
# import yaml

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
            name: str,
            low_freq: Dict,
            mid_freq: Dict,
            classification: Dict,
            # 共享特征提取器配置
             # 共享特征提取层
            # 分类头配置
            device: str = 'auto',
            init_seed: Optional[int] = None,
            evaluation: Dict = None
    ):
        super().__init__()
        self.name = name
        if device == 'auto':
            device = get_device()
        self.device = device
        per_label_hidden_dims: List[int] = classification['per_label_hidden_dims']  # 每个标签的分类头
        # 其他配置
        label_dropout: float = classification['label_dropout']  # 标签间的dropout
        num_labels = classification['num_labels']
        use_batch_norm = classification['use_batch_norm']

        self.num_labels = num_labels
        # 设置随机种子
        if init_seed is not None:
            self._set_init_seed(init_seed)

        print("=" * 70)
        print("构建多频段多标签二分类模型")
        print(f"标签数量: {num_labels}")
        print("=" * 70)

        # ===== 1. 低频特征提取器 =====
        print("\n1. 低频特征提取器")
        low_freq['device'] = device
        self.lf_extractor = AdvancedDNNCausalCNNRNNParallelWithFiLM(
            **low_freq,
        )

        # ===== 2. 中频特征提取器（使用FiLM） =====
        print("\n2. 中频特征提取器（使用FiLM调制）")

        self.mf_extractor = AdvancedDNNCausalCNNRNNParallelWithFiLM(
            **mid_freq,
            device=device
        )


        prev_dim = self.lf_extractor.out_puts[-1] + self.mf_extractor.out_puts[-1]


        # ===== 4. 多标签分类头 =====
        print("\n4. 多标签分类头")
        print(f"   每个标签独立分类头")
        print(f"   每个头隐藏层: {per_label_hidden_dims}")
        print(f"   输出: {num_labels}个独立的二分类概率")

        self.label_heads = nn.ModuleList()

        for label_idx in range(num_labels):
            # 每个标签的独立分类器
            label_layers = []
            label_prev_dim = prev_dim

            # 隐藏层
            for i, hidden_dim in enumerate(per_label_hidden_dims):
                linear = nn.Linear(label_prev_dim, hidden_dim)
                if i == 0:  # 第一层
                    nn.init.kaiming_normal_(linear.weight,
                                            mode='fan_in',
                                            nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_uniform_(linear.weight, gain=0.5)
                nn.init.zeros_(linear.bias)
                label_layers.append(linear)

                label_layers.append(nn.LeakyReLU(0.01))
                if use_batch_norm:
                    label_layers.append(nn.BatchNorm1d(hidden_dim))


                # 标签间dropout（防止标签间过度依赖）
                if label_dropout > 0:
                    label_layers.append(nn.Dropout(label_dropout))

                label_prev_dim = hidden_dim

            # 输出层：二分类（1个神经元）
            output_layer = nn.Linear(label_prev_dim, 1)
            nn.init.xavier_uniform_(output_layer.weight, gain=0.5)
            nn.init.zeros_(output_layer.bias)
            label_layers.append(output_layer)
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

        # 5. 多标签预测（每个标签独立）
        all_predictions = []
        for label_idx in range(self.num_labels):
            # 通过对应的标签头
            label_pred = self.label_heads[label_idx](combined)  # (B, 1)
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

