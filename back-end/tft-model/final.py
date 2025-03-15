import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset class including static features
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dynamic, data_static, targets):
        self.data_dynamic = torch.tensor(data_dynamic, dtype=torch.float32)
        self.data_static = torch.tensor(data_static, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data_dynamic)

    def __getitem__(self, idx):
        return self.data_dynamic[idx], self.data_static[idx], self.targets[idx]

# Gated Residual Network with LayerNorm and context
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_dim=0, dropout=0.1):
        super().__init__()
        self.context_dim = context_dim

        # Layers
        self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_dim, output_dim)
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)

        eta2 = self.elu(self.fc1(x))
        eta1 = self.dropout(self.fc2(eta2))

        gate = self.sigmoid(self.gate(eta1))

        if self.residual_proj is not None:
            residual = self.residual_proj(x[:, :self.residual_proj.in_features])
        else:
            residual = x

        output = self.layer_norm(residual + gate * eta1)
        return output

# Variable Selection Network
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, context_dim=0, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        self.grn = GatedResidualNetwork(
            input_dim=seq_len * input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            context_dim=context_dim,
            dropout=dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        batch_size, seq_len, input_dim = x.shape
        flattened = x.view(batch_size, -1)
        weights = self.grn(flattened, context)
        weights = self.softmax(weights)
        weighted_x = x * weights.unsqueeze(1)
        return weighted_x

# Interpretable Multi-Head Attention
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_transform = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.k_transform = nn.Linear(embed_dim, embed_dim)
        self.v_transform = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        key = self.k_transform(key)
        value = self.v_transform(value)

        head_outputs = []
        for i in range(self.num_heads):
            q = self.q_transform[i](query)
            attn = torch.matmul(q, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attn = self.dropout(torch.softmax(attn, dim=-1))
            head = torch.matmul(attn, value)
            head_outputs.append(head)

        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output(concatenated)
        return output, attn

# Temporal Fusion Transformer Model
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, static_size, hidden_size, num_heads, num_layers, output_size, seq_len, dropout=0.1):
        super().__init__()
        self.static_encoder = GatedResidualNetwork(static_size, hidden_size, hidden_size, dropout=dropout)
        self.var_select = VariableSelectionNetwork(input_size, seq_len, hidden_size, hidden_size, dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, output_size, dropout=dropout)

    def forward(self, x, static):
        static_context = self.static_encoder(static)
        selected = self.var_select(x, static_context)
        lstm_out, _ = self.lstm(selected)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        gate = self.sigmoid(self.gate(attn_out))
        gated = gate * attn_out + (1 - gate) * lstm_out
        output = self.output_grn(gated[:, -1, :])
        return output

# Data preprocessing with static feature
def preprocess_data(data):
    data = add_technical_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']
    data = data[features].dropna()
    data['DummyStatic'] = 0.0  # Add dummy static feature

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler.pkl')
    return scaled[:, :-1], scaled[:, -1], scaler  # Separate dynamic and static

# Create sequences with static feature
def create_sequences(dynamic_data, static_data, seq_length):
    xs, ys, statics = [], [], []
    for i in range(len(dynamic_data) - seq_length):
        xs.append(dynamic_data[i:i+seq_length])
        ys.append(dynamic_data[i+seq_length, 3])  # Close price
        statics.append(static_data[i+seq_length])
    return np.array(xs), np.array(ys), np.array(statics)

# Training function with static data
def train_model