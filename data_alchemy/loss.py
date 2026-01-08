import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

all_counts = {
    "5m": [1914733,  39699907, 17134392, 23835387, 1569181],
    "15m": [3010278, 39972498, 8721874,  29856783, 2592167],
    "30m": [4015274, 39536145, 5514332,  31598614, 3489235],
    "60m": [5339170, 38460596, 3456064,  32089240, 4808530],
    "180m": [7658287,35795872, 4238391,  30185961, 6275089],
}

all_counts_24 = {
    "5m":   [748879,  17601981, 7110024, 10615365, 672551],
    "15m":  [1194457, 17785457, 3337759, 13329602, 1101525],
    "30m":  [1582353, 17597543, 1952860,  14139627, 1476417],
    "60m":  [2142288, 17038478, 1088016,  14459609, 2020409],
    "180m": [3167595, 15760421, 1580027,  13616038, 2624719],
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


def multi_scale_classification_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale_names: Optional[List[str]] = None,
        num_classes_per_task: int = 5,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    仅用于训练的轻量版损失函数。
    不返回 details，不 detach，不构造 dict，最大化训练效率。
    """
    current_offset = 0
    total_loss = 0.0

    for i,  task_name_ls in enumerate(scale_names):
        # ==== LS Choice ====
        ls_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        ls_label = labels[:, i].long()
        weight_ls = class_weights.get(task_name_ls, None) if class_weights else None
        loss = F.cross_entropy(ls_logits, ls_label, weight=weight_ls, reduction='mean')
        total_loss += loss
        current_offset += num_classes_per_task

        # ==== Volatility ====
        '''
        task_name_vol = f"{scale}_vol"
        vol_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        vol_label = labels[:, 2 * i + 1].long()
        weight_vol = class_weights.get(task_name_vol, None) if class_weights else None
        loss_vol = F.cross_entropy(vol_logits, vol_label, weight=weight_vol, reduction='mean')
        total_loss += loss_vol * scale_w * vol_weight_factor
        current_offset += num_classes_per_task
        '''

    avg_loss = total_loss / (1.5 * len(scale_names))
    return avg_loss


def multi_scale_classification_loss_with_details(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale_names: Optional[List[str]] = None,
        num_classes_per_task: int = 5,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    用于验证/测试的详细版损失函数。
    返回总 loss、各子任务 loss、以及每个子任务的 (logits, labels) 用于 metric 计算。
    """

    current_offset = 0
    total_loss = 0.0
    loss_details = {}

    for i, task_name_ls in enumerate(scale_names):
        # ==== LS Choice ====
        ls_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        ls_label = labels[:,i].long()
        weight_ls = class_weights.get(task_name_ls, None) if class_weights else None
        loss_ls = F.cross_entropy(ls_logits, ls_label, weight=weight_ls, reduction='mean')
        loss_details[task_name_ls] = loss_ls.item()
        total_loss += loss_ls
        current_offset += num_classes_per_task

        # ==== Volatility ====
        '''
        task_name_vol = f"{scale}_vol"
        vol_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        vol_label = labels[:, 2 * i + 1].long()
        weight_vol = class_weights.get(task_name_vol, None) if class_weights else None
        loss_vol = F.cross_entropy(vol_logits, vol_label, weight=weight_vol, reduction='mean')
        loss_details[task_name_vol] = loss_vol.item()
        total_loss += loss_vol * scale_w * vol_weight_factor
        current_offset += num_classes_per_task
        '''

    avg_loss = total_loss / (1.5 * len(scale_names))
    return avg_loss, loss_details