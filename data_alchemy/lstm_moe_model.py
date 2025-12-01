# lstm_moe_model.py （优化版，兼容 ONNX）
import torch
import torch.nn as nn
from typing import Dict, Any


class LSTMMoEModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.total_dim = cfg['total_input_dim']
        self.main_dim = cfg['main_dim']
        self.lf_blocks = cfg['num_low_freq_blocks']
        self.block_size = cfg['low_freq_block_size']
        self.seq_len = cfg['seq_len']
        self.num_experts = cfg['num_experts']
        self.output_targets = cfg['output_targets']

        # Router LSTMs
        self.router_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.block_size,
                hidden_size=cfg['router_lstm_hidden'],
                batch_first=True,
                bidirectional=False
            )
            for _ in range(self.lf_blocks)
        ])

        router_input_dim = self.lf_blocks * cfg['router_lstm_hidden']
        self.router_net = nn.Sequential(
            nn.Linear(router_input_dim, cfg['router_output_dim']),
            nn.ReLU(),
            nn.Linear(cfg['router_output_dim'], self.num_experts),
            nn.Softmax(dim=-1)
        )

        # Experts
        self.experts = nn.ModuleList([
            nn.LSTM(
                input_size=self.main_dim,
                hidden_size=cfg['expert_lstm_hidden'],
                num_layers=cfg['expert_lstm_layers'],
                batch_first=True,
                dropout=cfg['expert_dropout'] if cfg['expert_lstm_layers'] > 1 else 0.0,
                bidirectional=False
            )
            for _ in range(self.num_experts)
        ])

        self.expert_heads = nn.ModuleList([
            nn.Linear(cfg['expert_lstm_hidden'], self.output_targets)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.total_dim and T == self.seq_len
        # 不需要，已经好了
        # x = x.transpose(1, 2)  # (B, T, D)

        x_main = x[:, :, :self.main_dim]
        x_low_start = self.main_dim

        router_hiddens = []
        for i in range(self.lf_blocks):
            start = x_low_start + i * self.block_size
            end = start + self.block_size
            block = x[:, :, start:end]

            _, (h_n, _) = self.router_lstms[i](block)
            # 使用 h_n[-1] 更安全（即使多层）
            h_last = h_n[-1]  # (B, hidden)
            router_hiddens.append(h_last)

        router_cond = torch.cat(router_hiddens, dim=-1)
        router_weights = self.router_net(router_cond)

        expert_outputs = []
        for i in range(self.num_experts):
            output_seq, _ = self.experts[i](x_main)
            last_output = output_seq[:, -1, :]
            pred = self.expert_heads[i](last_output)
            expert_outputs.append(pred.unsqueeze(-1))

        stacked = torch.cat(expert_outputs, dim=-1)
        final_output = torch.sum(stacked * router_weights.unsqueeze(1), dim=-1)
        return final_output