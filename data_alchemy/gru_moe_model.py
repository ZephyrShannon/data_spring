# model.py

import torch
import torch.nn as nn
from typing import Tuple, List

class ParallelStackedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, parallelism_factor: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_per_slice = hidden_size // parallelism_factor
        self.total_hidden_size = self.hidden_size_per_slice * parallelism_factor
        self.num_layers = num_layers
        self.parallelism_factor = parallelism_factor
        self.dropout = dropout

        if hidden_size % parallelism_factor != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by parallelism_factor ({parallelism_factor})")

        # 使用 nn.GRU 替代 nn.LSTM
        self.gru_slices = nn.ModuleList([
            nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size_per_slice,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(parallelism_factor)
        ])

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        slice_outputs = []
        slice_hiddens = []  # GRU 只有 hidden state，没有 cell state

        for gru_slice in self.gru_slices:
            out, hn = gru_slice(x)  # ← GRU 返回 (output, hidden)，不是 (output, (h, c))
            slice_outputs.append(out)
            slice_hiddens.append(hn)

        final_output = torch.cat(slice_outputs, dim=2)      # (B, S, total_hidden)
        final_hidden = torch.cat(slice_hiddens, dim=2)      # (L, B, total_hidden)

        # GRU 没有 cell state，所以只返回 hidden
        return final_output, final_hidden  # 注意：不再返回 tuple of (h, c)

class LowFreqRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, num_experts: int):
        super().__init__()
        self.gru = ParallelStackedGRU(  # ← 改名
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        self.router_net = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)  # ← 不再解包 (h, c)
        last_layer_hidden = h_n[-1]  # (batch_size, hidden_size_total)
        weights = self.router_net(last_layer_hidden)
        return self.softmax(weights)

class MidFreqExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, output_dim: int):
        super().__init__()
        self.gru = ParallelStackedGRU(  # ← 改名
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)  # ← 忽略 hidden state
        projected_out = self.output_projection(gru_out)
        return projected_out


class HighFreqFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, prediction_horizon: int):
        super().__init__()
        self.gru = ParallelStackedGRU(  # ← 改名
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        self.predictor = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step_out = gru_out[:, -1, :]
        prediction = self.predictor(last_step_out)
        return prediction


class ThreeLayerMoE(nn.Module):
    def __init__(self, args):
        super(ThreeLayerMoE, self).__init__()
        self.high_freq_dim = args['high_freq_dim']
        self.mid_freq_dim = args['mid_freq_dim']

        # --- Layers ---
        self.router = LowFreqRouter(
            input_dim=args['low_freq_dim'],
            hidden_dim=args['router_hidden'],
            num_layers=args['router_layers'],
            parallelism_factor=args['router_parallelism'],
            num_experts=args['num_experts']
        )

        # --- Experts ---
        # Define an output dimension for experts if different from their hidden size
        # Often, experts output their hidden state size. Let's assume that for now.
        self.expert_output_dim = args['expert_hidden'] # Or a fixed/configurable value

        self.experts = nn.ModuleList([
            MidFreqExpert(
                input_dim=args['mid_freq_dim'],
                hidden_dim=args['expert_hidden'],
                num_layers=args['expert_layers'],
                parallelism_factor=args['expert_parallelism'],
                output_dim=self.expert_output_dim
            ) for _ in range(args['num_experts'])
        ])

        # --- Fusion ---
        # Calculate input dimension for fusion LSTM:
        # High-frequency features + weighted combination of expert outputs (each expert contributes expert_output_dim)
        fusion_input_dim = args['high_freq_dim'] +  self.expert_output_dim

        self.fusion = HighFreqFusion(
            input_dim=fusion_input_dim,
            hidden_dim=args['fusion_hidden'],
            num_layers=args['fusion_layers'],
            parallelism_factor=args['fusion_parallelism'],
            prediction_horizon=args['output_targets']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Assume x is already split according to args.{high,mid,low}_freq_dim ---
        # x shape: (batch_size, seq_len, total_input_dim)
        batch_size, seq_len, _ = x.shape

        # Split the input features
        x_high = x[:, :, :self.high_freq_dim]
        x_mid = x[:, :, self.high_freq_dim:self.high_freq_dim + self.mid_freq_dim]
        x_low = x[:, :, self.high_freq_dim + self.mid_freq_dim:]

        # --- 1. Low-Frequency Routing ---
        routing_weights = self.router(x_low) # (batch_size, num_experts)
        # print(f"Routing weights shape: {routing_weights.shape}")

        # --- 2. Mid-Frequency Expert Processing ---
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get output from expert for all timesteps
            exp_out = expert(x_mid) # (batch_size, seq_len, expert_output_dim)
            # Weight the entire expert output sequence by the corresponding routing weight
            # Expand weight to match dimensions for broadcasting
            weight = routing_weights[:, i].unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1)
            # print(f"Weight shape for expert {i}: {weight.shape}")
            weighted_exp_out = exp_out * weight # Broadcasting: (B, S, E_OD) * (B, 1, 1) -> (B, S, E_OD)
            expert_outputs.append(weighted_exp_out)

        # Sum the weighted outputs of all experts across the sequence
        # expert_outputs is a list of tensors [(B, S, E_OD), ...]
        combined_expert_output = torch.stack(expert_outputs, dim=-1).sum(dim=-1) # (B, S, E_OD)
        # Alternative sum: combined_expert_output = torch.sum(torch.stack(expert_outputs), dim=0)

        # --- 3. High-Frequency Fusion ---
        # Concatenate high-frequency input with the combined expert output
        fusion_input = torch.cat((x_high, combined_expert_output), dim=2) # (B, S, high_freq_dim + E_OD)
        # print(f"Fusion input shape: {fusion_input.shape}")

        # Pass through fusion LSTM and get final prediction
        prediction = self.fusion(fusion_input) # (batch_size, prediction_horizon)

        return prediction

# --- Example Usage ---
if __name__ == '__main__':
    # Import torch here as well if running this script directly
    import torch
    # Initialize the model with the configuration
    model = ThreeLayerMoE(args).to(args.device)

    # Create dummy input data matching the configured dimensions
    batch_size = args.batch_size
    seq_len = args.seq_len
    total_input_dim = args.total_input_dim
    dummy_input = torch.randn(batch_size, seq_len, total_input_dim).to(args.device)

    # --- Test with Parallelism ---
    print("--- Testing Model with Configurable Parallelism ---")
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Seq Len: {args.seq_len}")
    print(f"  Total Input Dim: {args.total_input_dim}")
    print(f"  High Freq Dim: {args.high_freq_dim}")
    print(f"  Mid Freq Dim: {args.mid_freq_dim}")
    print(f"  Low Freq Dim: {args.low_freq_dim}")
    print(f"  Num Experts: {args.num_experts}")
    print(f"  Router LSTM: H={args.router_lstm_hidden}, L={args.router_lstm_layers}, P={args.router_lstm_parallelism}")
    print(f"  Expert LSTM: H={args.expert_mid_lstm_hidden}, L={args.expert_mid_lstm_layers}, P={args.expert_mid_lstm_parallelism}")
    print(f"  Fusion LSTM: H={args.fusion_lstm_hidden}, L={args.fusion_lstm_layers}, P={args.fusion_lstm_parallelism}")
    print("-" * 20)

    try:
        model.eval() # Set to evaluation mode to avoid issues like Dropout if present
        with torch.no_grad(): # Disable gradient calculation for dummy inference
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}") # Should be (batch_size, prediction_horizon)
            print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
