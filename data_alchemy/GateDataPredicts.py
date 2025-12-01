import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# 1. LSTM Expert
# ======================
class LSTMExpert(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=32, num_layers=1, dropout=0.1):
        super(LSTMExpert, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        h_last = hidden[-1]
        return self.fc(self.dropout(h_last))


# ======================
# 2. Router Network
# ======================
class Router(nn.Module):
    def __init__(self, low_freq_dim=18, hidden_dim=32, num_experts=8, dropout=0.1):
        super(Router, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(low_freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


# ======================
# 3. Conditional MoE
# ======================
class ConditionalMoE(nn.Module):
    def __init__(self, input_dim=5, low_freq_dim=18, num_experts=8,
                 hidden_dim=64, output_dim=32, expert_dropout=0.1, router_dropout=0.1):
        super(ConditionalMoE, self).__init__()
        self.num_experts = num_experts

        # 专家列表
        self.experts = nn.ModuleList([
            LSTMExpert(input_dim, hidden_dim, output_dim, num_layers=1, dropout=expert_dropout)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = Router(low_freq_dim, hidden_dim=32, num_experts=num_experts, dropout=router_dropout)

    def forward(self, high_freq_x, low_freq_x):
        # 路由权重
        weights = self.router(low_freq_x)  # (B, 8)
        weights_expanded = weights.unsqueeze(-1)  # (B, 8, 1)

        # 所有专家并行处理
        expert_outputs = torch.stack([expert(high_freq_x) for expert in self.experts], dim=1)  # (B, 8, 32)

        # 加权融合
        output = torch.sum(expert_outputs * weights_expanded, dim=1)  # (B, 32)

        return output, weights  # 返回特征和路由权重