import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Variable Selection Network
        self.variable_selection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Variable selection
        variable_weights = self.variable_selection(x)
        x = x * variable_weights

        # LSTM for temporal processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        # Multi-head attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output)

        # Gating mechanism
        gate_output = self.gate(attn_output)
        gated_output = gate_output * attn_output

        # Final output
        output = self.fc(gated_output[:, -1, :])
        return output