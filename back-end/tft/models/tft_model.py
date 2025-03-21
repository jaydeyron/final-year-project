import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal data
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PriceConstraintLayer(nn.Module):
    """
    Custom layer to constrain predictions to be within a reasonable range
    of the last observed price
    """
    def __init__(self, max_change_percent=0.05):
        super(PriceConstraintLayer, self).__init__()
        self.max_change_percent = max_change_percent
        
    def forward(self, x, last_close):
        # Calculate allowable price range
        min_price = last_close * (1 - self.max_change_percent)
        max_price = last_close * (1 + self.max_change_percent)
        
        # Instead of hard clamping, use a soft constraint
        # This adds a penalty but allows exceeding the threshold if the model is confident
        x_constrained = x.clone()
        above_max = x > max_price
        below_min = x < min_price
        
        # Apply soft constraints
        x_constrained[above_max] = max_price + 0.2 * (x[above_max] - max_price)
        x_constrained[below_min] = min_price + 0.2 * (x[below_min] - min_price)
        
        return x_constrained

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, dropout=0.1, 
                 max_change_percent=0.05):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_change_percent = max_change_percent
        
        # Feature processing layers
        self.feature_layer = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention transformer layer for variable selection
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers
        self.pre_output = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Constrained output
        self.price_constraint = PriceConstraintLayer(max_change_percent)
        
    def extract_last_close_price(self, x):
        # Extract last close price from each sequence (assuming close price is at index 3)
        # Shape: [batch_size, seq_len, features] -> [batch_size, 1]
        return x[:, -1, 3:4]  # Last timestep's close price
        
    def forward(self, x, apply_constraint=False):
        # x shape: [batch_size, sequence_length, input_size]
        batch_size, seq_len, _ = x.shape
        
        # Store last close price for constraint
        last_close = self.extract_last_close_price(x)
        
        # Feature transformation
        x = self.feature_layer(x)
        x = self.positional_encoding(x)
        
        # Temporal processing
        x, _ = self.lstm(x)
        
        # Self-attention mechanism
        x = self.transformer_encoder(x)
        
        # Take the last output for prediction
        x = x[:, -1, :]
        
        # Output layers with residual connection
        x = F.relu(self.pre_output(x)) + x
        output = self.output_layer(x)
        
        # Apply price constraint only if explicitly requested
        # Set default to False to see raw model predictions
        if apply_constraint:
            output = self.price_constraint(output, last_close)
        
        return output