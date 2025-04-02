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
        # Make constraints more adaptable to market conditions
        # Instead of fixed percentages, make the allowed range dependent on recent volatility
        batch_size = x.size(0)
        
        # Allow wider range for more confident predictions
        confidence_factor = torch.sigmoid(torch.abs(x - last_close) / (last_close * 0.01))
        
        # Dynamic constraint based on confidence
        effective_constraint = self.max_change_percent * (1.0 + confidence_factor)
        
        # Calculate allowable price range
        min_price = last_close * (1 - effective_constraint)
        max_price = last_close * (1 + effective_constraint)
        
        # Apply soft constraints with smoother transition
        x_constrained = torch.where(
            x > max_price,
            max_price + 0.3 * torch.tanh((x - max_price) / (last_close * 0.01)),
            x
        )
        
        x_constrained = torch.where(
            x_constrained < min_price,
            min_price + 0.3 * torch.tanh((x_constrained - min_price) / (last_close * 0.01)),
            x_constrained
        )
        
        return x_constrained

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, dropout=0.1, 
                 max_change_percent=0.05):
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_change_percent = max_change_percent
        
        # Enhanced Feature processing layers with variable importance
        self.feature_layer = nn.Linear(input_size, hidden_size)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        # Improved temporal processing with bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional for better context
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Reduce dimensionality back after bidirectional
        self.lstm_reducer = nn.Linear(hidden_size * 2, hidden_size)
        
        # Add temporal attention to focus on important timepoints
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
        
        # Enhanced output layers with skip connections
        self.pre_output_1 = nn.Linear(hidden_size, hidden_size)
        self.pre_output_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Uncertainty estimation - predict both mean and variance
        self.uncertainty_layer = nn.Linear(hidden_size, output_size)
        
        # Constrained output
        self.price_constraint = PriceConstraintLayer(max_change_percent)
        
        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def extract_last_close_price(self, x):
        # Extract last close price from each sequence (assuming close price is at index 3)
        return x[:, -1, 3:4]  # Last timestep's close price
        
    def forward(self, x, apply_constraint=False):
        batch_size, seq_len, _ = x.shape
        
        # Store last close price for constraint
        last_close = self.extract_last_close_price(x)
        
        # Enhanced feature processing with variable importance weighting
        feature_weights = self.feature_gate(x)
        x = x * feature_weights
        
        # Feature transformation
        x = self.feature_layer(x)
        x = self.positional_encoding(x)
        
        # Temporal processing with bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        # Reduce dimensionality back to hidden_size
        lstm_out = self.lstm_reducer(lstm_out)
        lstm_out = F.relu(lstm_out)
        
        # Apply layer normalization
        lstm_out = self.layer_norm1(lstm_out)
        
        # Apply temporal self-attention
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = lstm_out + self.dropout(attn_out)  # Residual connection
        lstm_out = self.layer_norm2(lstm_out)
        
        # Self-attention mechanism with transformer
        transformer_out = self.transformer_encoder(lstm_out)
        
        # Take weighted combination of outputs for dynamic horizon attention
        # This helps the model focus on the most relevant past timestamps
        attention_weights = F.softmax(torch.matmul(
            transformer_out[:, -1:, :],  # Use last timestep as query
            transformer_out.transpose(1, 2)  # Key = all timesteps
        ), dim=-1)
        
        context_vector = torch.matmul(attention_weights, transformer_out).squeeze(1)
        
        # Add the last timestamp representation with context
        final_repr = transformer_out[:, -1, :] + 0.3 * context_vector
        final_repr = self.layer_norm3(final_repr)
        
        # Multi-layer output with residual connections
        hidden1 = F.relu(self.pre_output_1(final_repr))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.pre_output_2(hidden1))
        hidden2 = self.dropout(hidden2)
        
        # Residual connection to improve gradient flow
        output_features = hidden2 + 0.2 * hidden1 + 0.1 * final_repr
        
        # Main prediction
        output = self.output_layer(output_features)
        
        # Estimate uncertainty (not used directly in prediction, but useful for model development)
        uncertainty = F.softplus(self.uncertainty_layer(output_features))
        
        # Apply price constraint only if explicitly requested
        if apply_constraint:
            output = self.price_constraint(output, last_close)
        
        return output