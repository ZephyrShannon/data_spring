# model.py

import torch
import torch.nn as nn
from typing import Tuple, List
from config import args # Import the configuration

class ParallelStackedLSTM(nn.Module):
    """A module representing a single LSTM block potentially split into parallel slices."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, parallelism_factor: int, dropout: float = 0.0):
        super(ParallelStackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size_per_slice = hidden_size // parallelism_factor
        self.num_layers = num_layers
        self.parallelism_factor = parallelism_factor
        self.dropout = dropout

        if parallelism_factor <= 0:
            raise ValueError("parallelism_factor must be positive.")

        # Create 'parallelism_factor' number of LSTM stacks
        self.lstm_slices = nn.ModuleList([
            nn.LSTM(
                input_size=self.input_size if layer_idx == 0 else self.hidden_size_per_slice * parallelism_factor,
                hidden_size=self.hidden_size_per_slice,
                num_layers=num_layers,
                batch_first=True,
                dropout=self.dropout if num_layers > 1 else 0.0 # Dropout between layers
            )
            for _ in range(parallelism_factor)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the parallel LSTM stacks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output features from the last layer for all timesteps:
                  Shape (batch_size, seq_len, hidden_size_per_slice * parallelism_factor).
                - Tuple containing:
                    - Final hidden state (h_n): Shape (num_layers, batch_size, hidden_size_per_slice * parallelism_factor).
                    - Final cell state (c_n): Shape (num_layers, batch_size, hidden_size_per_slice * parallelism_factor).
        """
        batch_size, seq_len, _ = x.shape

        # Lists to store outputs and final states from each slice
        slice_outputs = []
        slice_hiddens = []
        slice_cells = []

        # Process each parallel slice
        for lstm_slice in self.lstm_slices:
            # Pass input through the current LSTM slice
            slice_out, (slice_hn, slice_cn) = lstm_slice(x)
            slice_outputs.append(slice_out)
            slice_hiddens.append(slice_hn) # Shape: (num_layers, batch_size, hidden_size_per_slice)
            slice_cells.append(slice_cn)   # Shape: (num_layers, batch_size, hidden_size_per_slice)

            # Update input for the next layer within the same slice
            # In a standard stacked LSTM, output of one layer feeds into the next.
            # Here, since slices are independent, we keep the original input for simplicity
            # or pass the output of the previous slice if chaining is intended differently.
            # The current structure treats slices as parallel stacks of full 'num_layers' LSTMs.
            # If you want intra-slice stacking with inter-slice connections, it would be more complex.
            # This implementation assumes parallel, independent multi-layer stacks.

        # Concatenate outputs along the feature dimension
        # Output shape: (batch_size, seq_len, hidden_size_per_slice * parallelism_factor)
        final_output = torch.cat(slice_outputs, dim=2)

        # Concatenate final hidden and cell states along the feature dimension
        # State shape: (num_layers, batch_size, hidden_size_per_slice * parallelism_factor)
        final_hidden = torch.cat(slice_hiddens, dim=2)
        final_cell = torch.cat(slice_cells, dim=2)

        return final_output, (final_hidden, final_cell)


class LowFreqRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, num_experts: int):
        super(LowFreqRouter, self).__init__()
        self.lstm = ParallelStackedLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        # Network to compute expert weights from the final LSTM hidden state
        # Uses the potentially reduced effective hidden size from concatenation
        self.router_net = nn.Linear(hidden_dim, num_experts) # hidden_dim should be adjusted if needed
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        # Use the hidden state from the last layer (index -1) and last time step (squeeze if needed, but h_n is [layers, B, H])
        # h_n shape: (num_layers, batch_size, hidden_size_total)
        last_layer_hidden = h_n[-1] # Shape: (batch_size, hidden_size_total)
        weights = self.router_net(last_layer_hidden)
        return self.softmax(weights)


class MidFreqExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, output_dim: int):
        super(MidFreqExpert, self).__init__()
        self.lstm = ParallelStackedLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        # Project LSTM output to desired intermediate output dimension
        # Could be hidden_dim, or a different size if needed before fusion
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get output for all timesteps
        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, hidden_size_total)
        projected_out = self.output_projection(lstm_out) # (batch_size, seq_len, output_dim)
        return projected_out


class HighFreqFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, parallelism_factor: int, prediction_horizon: int):
        super(HighFreqFusion, self).__init__()
        self.lstm = ParallelStackedLSTM(
            input_size=input_dim, # This will be high_freq_dim + num_experts * expert_output_dim
            hidden_size=hidden_dim,
            num_layers=num_layers,
            parallelism_factor=parallelism_factor
        )
        # Final projection to prediction horizon
        self.predictor = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, hidden_size_total)
        # Typically, use the output from the last time step for final prediction
        last_step_out = lstm_out[:, -1, :] # (batch_size, hidden_size_total)
        prediction = self.predictor(last_step_out) # (batch_size, prediction_horizon)
        return prediction


class ThreeLayerMoE(nn.Module):
    def __init__(self, args):
        super(ThreeLayerMoE, self).__init__()
        self.args = args

        # --- Layers ---
        self.router = LowFreqRouter(
            input_dim=args.low_freq_dim,
            hidden_size=args.router_lstm_hidden,
            num_layers=args.router_lstm_layers,
            parallelism_factor=args.router_lstm_parallelism,
            num_experts=args.num_experts
        )

        # --- Experts ---
        # Define an output dimension for experts if different from their hidden size
        # Often, experts output their hidden state size. Let's assume that for now.
        self.expert_output_dim = args.expert_mid_lstm_hidden # Or a fixed/configurable value

        self.experts = nn.ModuleList([
            MidFreqExpert(
                input_dim=args.mid_freq_dim,
                hidden_dim=args.expert_mid_lstm_hidden,
                num_layers=args.expert_mid_lstm_layers,
                parallelism_factor=args.expert_mid_lstm_parallelism,
                output_dim=self.expert_output_dim
            ) for _ in range(args.num_experts)
        ])

        # --- Fusion ---
        # Calculate input dimension for fusion LSTM:
        # High-frequency features + weighted combination of expert outputs (each expert contributes expert_output_dim)
        fusion_input_dim = args.high_freq_dim + args.num_experts * self.expert_output_dim

        self.fusion = HighFreqFusion(
            input_dim=fusion_input_dim,
            hidden_size=args.fusion_lstm_hidden,
            num_layers=args.fusion_lstm_layers,
            parallelism_factor=args.fusion_lstm_parallelism,
            prediction_horizon=args.prediction_horizon
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Assume x is already split according to args.{high,mid,low}_freq_dim ---
        # x shape: (batch_size, seq_len, total_input_dim)
        batch_size, seq_len, _ = x.shape

        # Split the input features
        x_high = x[:, :, :self.args.high_freq_dim]
        x_mid = x[:, :, self.args.high_freq_dim:self.args.high_freq_dim + self.args.mid_freq_dim]
        x_low = x[:, :, self.args.high_freq_dim + self.args.mid_freq_dim:]

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
