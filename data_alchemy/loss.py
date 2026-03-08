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
    Multi-head binary focal loss.

    Each head is an independent binary classification task.

    Args:
        alphas: List[float] of length num_heads.
                alpha for each head (weight for positive class).
                If None, use 0.8 for all heads.
        gamma: Focusing parameter (default=2.0)
        reduction: 'mean' or 'sum' over all heads and samples
    """

    def __init__(self, alphas: Optional[List[float]]):
        super().__init__()
        self.alphas = alphas

        if alphas is not None:
            self.alphas = torch.tensor(alphas, dtype=torch.float32)  # (H,)
            self.alphas = self.alphas/(1-self.alphas)
        else:
            raise Exception("alphas is None!") # will be handled in forward

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            reduction='none'  # 保持每个元素的损失
        )

        # 2. 创建放大因子
        # 正样本：alpha倍
        # 负样本：1倍
        multiplier = torch.where(
            targets == 1,
            self.alphas.clone().detach().to(logits.device),
            torch.tensor(1., device=logits.device)
        )
        amplified_loss = base_loss * multiplier
        return amplified_loss.mean()


    @staticmethod
    def compute_metrics_per_head(
            preds: torch.Tensor,
            targets: torch.Tensor
    ) -> List[Tuple[float, float, float]]:
        """
        Compute Precision, Recall, F1 for each binary classification head.

        Args:
            preds: (B, H) —— predicted 0/1 labels
            targets: (B, H) —— true 0/1 labels

        Returns:
            List of (precision, recall, f1) for each head (length = H)
        """
        B, H = preds.shape
        assert targets.shape == (B, H)

        metrics = []
        probabilities = torch.sigmoid(preds)
        # 转换为二进制预测
        preds = (probabilities >= 0.5)
        targets = targets.bool()

        for h in range(H):
            tp = (preds[:, h] & targets[:, h]).sum().item()
            fp = (preds[:, h] & ~targets[:, h]).sum().item()
            fn = (~preds[:, h] & targets[:, h]).sum().item()
            print(f"tp={tp}, fp={fp}, fn={fn}")
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics.append((precision, recall, f1))

        return metrics
