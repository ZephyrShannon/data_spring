import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


def create_fixed_class_weights(
        scales: List[str] = ['5m', '15m', '30m', '60m', '180m'],
        raw_weights: List[float] = [2, 1, 0.5, 1, 2],
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
        class_weights[f"{scale}_ls"] = weight_tensor.clone()
        class_weights[f"{scale}_vol"] = weight_tensor.clone()

    return class_weights


def multi_scale_classification_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale_weights: Optional[List[Tuple[str, float]]] = None,
        vol_weight_factor: float = 0.6,
        num_classes_per_task: int = 5,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    仅用于训练的轻量版损失函数。
    不返回 details，不 detach，不构造 dict，最大化训练效率。
    """

    current_offset = 0
    total_loss = 0.0

    for i, (scale, scale_w) in enumerate(scale_weights):
        # ==== LS Choice ====
        task_name_ls = f"{scale}_ls"
        ls_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        ls_label = labels[:, 2 * i].long()
        weight_ls = class_weights.get(task_name_ls, None) if class_weights else None
        loss_ls = F.cross_entropy(ls_logits, ls_label, weight=weight_ls, reduction='mean')
        total_loss += loss_ls * scale_w
        current_offset += num_classes_per_task

        # ==== Volatility ====
        task_name_vol = f"{scale}_vol"
        vol_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        vol_label = labels[:, 2 * i + 1].long()
        weight_vol = class_weights.get(task_name_vol, None) if class_weights else None
        loss_vol = F.cross_entropy(vol_logits, vol_label, weight=weight_vol, reduction='mean')
        total_loss += loss_vol * scale_w * vol_weight_factor
        current_offset += num_classes_per_task

    avg_loss = total_loss / (1.5 * len(scale_weights))
    return avg_loss


def multi_scale_classification_loss_with_details(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scale_weights: Optional[List[Tuple[str, float]]] = None,
        vol_weight_factor: float = 0.5,
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

    for i, (scale, scale_w) in enumerate(scale_weights):
        # ==== LS Choice ====
        task_name_ls = f"{scale}_ls"
        ls_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        ls_label = labels[:, 2 * i].long()
        weight_ls = class_weights.get(task_name_ls, None) if class_weights else None
        loss_ls = F.cross_entropy(ls_logits, ls_label, weight=weight_ls, reduction='mean')
        loss_details[task_name_ls] = loss_ls.item()
        total_loss += loss_ls * scale_w
        current_offset += num_classes_per_task

        # ==== Volatility ====
        task_name_vol = f"{scale}_vol"
        vol_logits = logits[:, current_offset:current_offset + num_classes_per_task]
        vol_label = labels[:, 2 * i + 1].long()
        weight_vol = class_weights.get(task_name_vol, None) if class_weights else None
        loss_vol = F.cross_entropy(vol_logits, vol_label, weight=weight_vol, reduction='mean')
        loss_details[task_name_vol] = loss_vol.item()
        total_loss += loss_vol * scale_w * vol_weight_factor
        current_offset += num_classes_per_task

    avg_loss = total_loss / (2 * len(scale_weights))
    return avg_loss, loss_details