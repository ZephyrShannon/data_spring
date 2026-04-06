import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

all_counts = {
    "5m": [1914733, 39699907, 17134392, 23835387, 1569181],
    "15m": [3010278, 39972498, 8721874, 29856783, 2592167],
    "30m": [4015274, 39536145, 5514332, 31598614, 3489235],
    "60m": [5339170, 38460596, 3456064, 32089240, 4808530],
    "180m": [7658287, 35795872, 4238391, 30185961, 6275089],
}

all_counts_24 = {
    "5m": [748879, 17601981, 7110024, 10615365, 672551],
    "15m": [1194457, 17785457, 3337759, 13329602, 1101525],
    "30m": [1582353, 17597543, 1952860, 14139627, 1476417],
    "60m": [2142288, 17038478, 1088016, 14459609, 2020409],
    "180m": [3167595, 15760421, 1580027, 13616038, 2624719],
}


def compute_class_weights_for_cross_entropy(counts, adjust_factor, scale="5m"):
    """
    根据指定时间尺度的类别计数，生成用于 CrossEntropyLoss 的 weight。

    Args:
        counts (dict): 包含各时间尺度类别计数的字典
        scale (str): 时间尺度，如 "5m", "15m" 等

    Returns:
        torch.Tensor: shape (5,), dtype=float32，可直接传给 nn.CrossEntropyLoss(weight=...)
    """
    # 获取该尺度下的类别计数
    class_counts = np.array(counts[scale])  # shape (5,)

    # 避免除零（虽然你的数据不会为0）
    assert np.all(class_counts > 0), "All class counts must be positive!"

    # 基础权重：频率的倒数（越少的类，权重越大）
    base_weights = 1.0 / class_counts

    # 应用自定义调整因子
    adjustment_factors = np.array(adjust_factor)  # idx 0～4

    adjusted_weights = base_weights * adjustment_factors

    # 可选：归一化（非必须，CrossEntropyLoss 不要求 weight 归一化）
    # 但有时为了数值稳定或便于比较，可除以均值
    adjusted_weights = adjusted_weights / adjusted_weights.mean()

    return torch.tensor(adjusted_weights, dtype=torch.float32)


def create_fixed_class_weights(
        scales: List[str] = ['5m', '15m', '30m', '60m', '180m'],
        raw_weights: List[float] = [1.5, 1, 0.7, 1, 1.5],
        normalize_to_mean_one: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    为每个子任务创建相同的固定类别权重。

    Args:
        scales: 时间尺度列表
        raw_weights: 人工指定的原始权重（长度=5）
        normalize_to_mean_one: 是否将权重缩放至均值为1

    Returns:
        dict like {'5m_ls': tensor([...]), '5m_vol': tensor([...]), ...}
    """
    if normalize_to_mean_one:
        mean_w = sum(raw_weights) / len(raw_weights)
        weights = [w / mean_w for w in raw_weights]
    else:
        weights = raw_weights

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    class_weights = {}

    for scale in scales:
        class_weights[f"{scale}_ls"] = compute_class_weights_for_cross_entropy(all_counts_24, raw_weights, scale)
        class_weights[f"{scale}_vol"] = weight_tensor.clone()

    return class_weights




def compute_focal_loss_params(
        num_pos: int,
        num_neg: int,
        gamma: float = 2.0
) -> dict:
    """
    根据正负样本数量自动推荐 Focal Loss 参数

    Args:
        num_pos: 正样本数量
        num_neg: 负样本数量
        gamma: 聚焦参数，默认 2.0

    Returns:
        dict: {'alpha': float, 'gamma': float}
    """
    pos_ratio = num_pos / (num_pos + num_neg)

    if pos_ratio >= 0.1:
        alpha = 0.75
    elif pos_ratio >= 0.05:  # ～1:20
        alpha = 0.80
    elif pos_ratio >= 0.02:  # ～1:50
        alpha = 0.85
    else:
        alpha = 0.90

    return {"alpha": alpha, "gamma": gamma}


def compute_metrics(inputs, targets):
    """
    计算正分类的准确率、召回率和F1分数。

    参数:
    - inputs: 模型的logits (N,) 或者经过sigmoid激活后的概率值。
    - targets: 真实标签 (N,)。

    返回:
    - 正分类的准确率、召回率和F1分数。
    """
    # 将logits转换为预测概率
    probabilities = torch.sigmoid(inputs)
    # 转换为二进制预测
    predictions = (probabilities >= 0.5).float()

    # 将张量移动到CPU并转换为numpy数组进行评估
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # 计算指标
    accuracy = accuracy_score(targets_np, predictions_np)
    recall = recall_score(targets_np, predictions_np, pos_label=1)
    f1 = f1_score(targets_np, predictions_np)

    return accuracy, recall, f1


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (N,), targets: binary labels (N,)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MultiHeadBinaryFocalLoss(nn.Module):
    """
    多头 Focal Loss - 支持多标签/多任务分类

    特点:
    - 支持多个独立的二分类任务
    - 每个任务可以有独立的 alpha 和 gamma
    - 支持样本级和任务级的权重
    - 数值稳定，防止梯度爆炸
    """

    def __init__(
            self,
            num_heads: int,
            device,
            alphas,
            gamma: float = 2.0,
    ):
        """
        Args:
            num_heads: 任务数量（头数）
            alphas: 每个头的 alpha 参数
                    - None: 使用默认值 0.75
                    - float: 所有头使用相同的 alpha
                    - List[float]: 每个头独立的 alpha
            gamma: 聚焦参数，默认 2.0
            device: 设备
        """
        super().__init__()

        self.num_heads = num_heads
        self.gamma = gamma

        # 处理 alphas
        if alphas is None:
            # 默认所有头使用 0.75
            alphas = [0.8] * num_heads
        elif isinstance(alphas, (int, float)):
            alphas = [float(alphas)] * num_heads
        elif len(alphas) != num_heads:
            raise ValueError(f"alphas length {len(alphas)} != num_heads {num_heads}")

        # 注册为 buffer（可选的，这里用普通属性也可以）
        self.alphas = torch.tensor(alphas, dtype=torch.float32, device=device)

        # 预计算正负样本权重
        self.pos_weights = self.alphas
        self.neg_weights = 1.0 - self.alphas


    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算多头 Focal Loss

        Args:
            logits: (batch_size, num_heads) - 模型输出的 logits
            targets: (batch_size, num_heads) - 真实标签 (0 或 1)

        Returns:
            loss: 标量损失值
            components: (可选) 各个组件的字典
        """
        # 1. 输入验证
        assert logits.shape == targets.shape, \
            f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
        assert logits.size(1) == self.num_heads, \
            f"Expected {self.num_heads} heads, got {logits.size(1)}"

        # 2. 数值稳定处理
        logits = torch.clamp(logits, -10, 10)

        # 4. 计算 BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # (batch_size, num_heads)

        # 5. 计算 pt（模型对正确类的预测概率）
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, 1e-7, 1 - 1e-7)  # 防止极端值
        # 6. 计算 Focal weight
        focal_weight = (1 - pt) ** self.gamma  # (batch_size, num_heads)

        # 7. 计算 Alpha weight
        alpha_weight = torch.where(
            targets == 1,
            self.pos_weights.view(1, -1),
            self.neg_weights.view(1, -1)
        )  # (batch_size, num_heads)

        # 8. 组合权重
        total_weight = focal_weight * alpha_weight  # (batch_size, num_heads)
        # 10. 计算加权 loss
        weighted_loss = total_weight * bce_loss  # (batch_size, num_heads)

        return weighted_loss.mean()

    @staticmethod
    def compute_metrics_vectorized(logits, targets):
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1 - targets)).sum(dim=0)
        fn = ((1 - preds) * targets).sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    @staticmethod
    def compute_metrics_per_head(
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> List[Tuple[float, float, float]]:
        """
        Compute Precision, Recall, F1 for each binary classification head.

        Args:
            logits: (B, H) —— returns of the model
            targets: (B, H) —— true 0/1 labels

        Returns:
            List of (precision, recall, f1) for each head (length = H)
        """
        B, H = logits.shape
        assert targets.shape == (B, H)

        metrics = []
        probabilities = torch.sigmoid(logits)
        # 转换为二进制预测
        preds = (probabilities >= 0.5)
        targets = targets.bool()

        for h in range(H):
            pred_h = preds[:, h]
            target_h = targets[:, h]
            tp = ((pred_h == True) & (target_h == True)).sum().item()
            fp = ((pred_h == True) & (target_h == False)).sum().item()
            fn = ((pred_h == False) & (target_h == True)).sum().item()
            #print(f"tp={tp}, fp={fp}, fn={fn}")
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics.append((precision, recall, f1))

        return metrics
