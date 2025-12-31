import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def multi_scale_classification_loss_with_details(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_config: Dict[str, Dict[str, int]]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = logits.shape[0]
    scales = ['5m', '15m', '30m', '60m', '180m']
    current_offset = 0
    total_loss = 0.0
    details = {}

    for i, scale in enumerate(scales):
        # LS Choice
        ls_classes = class_config['ls_choice'][scale]
        ls_logits = logits[:, current_offset:current_offset + ls_classes]
        ls_label = labels[:, 2 * i].long()
        loss_ls = F.cross_entropy(ls_logits, ls_label, reduction='mean')
        details[f"{scale}_ls"] = loss_ls.item()
        total_loss += loss_ls
        current_offset += ls_classes

        # Volatility
        vol_classes = class_config['volat'][scale]
        vol_logits = logits[:, current_offset:current_offset + vol_classes]
        vol_label = labels[:, 2 * i + 1].long()
        loss_vol = F.cross_entropy(vol_logits, vol_label, reduction='mean')
        details[f"{scale}_vol"] = loss_vol.item()
        total_loss += loss_vol
        current_offset += vol_classes

    avg_loss = total_loss / (2 * len(scales))
    return avg_loss, details