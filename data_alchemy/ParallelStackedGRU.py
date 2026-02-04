# ==============================
# 2. ParallelStackedGRU (你的原始实现，复用)
# ==============================
import torch
from torch import nn


class ParallelStackedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        parallelism_factor: int,
        dropout: float = 0.0
    ):
        super().__init__()
        if hidden_size % parallelism_factor != 0:
            raise ValueError("hidden_size must be divisible by parallelism_factor")
        self.hidden_per_slice = hidden_size // parallelism_factor
        self.gru_slices = nn.ModuleList([
            nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_per_slice,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(parallelism_factor)
        ])

    def forward(self, x: torch.Tensor):
        outputs = []
        for gru in self.gru_slices:
            out, _ = gru(x)
            outputs.append(out)
        return torch.cat(outputs, dim=-1), None