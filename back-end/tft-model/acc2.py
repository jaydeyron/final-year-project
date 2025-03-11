import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Gated Residual Network
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layernorm(x + residual)
        return torch.sigmoid(self.gate(x)) * x + (1 - torch.sigmoid(self.gate(x))) * residual

# Temporal Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.multihead = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead(x, x, x)
        return self.layernorm(x + self.dropout(attn_output))

# Temporal Fusion Transformer
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout=0.3):
        super().__init__()
        self.grn = GatedResidualNetwork(input_dim, hidden_dim, dropout)
        self.temporal_attention = nn.ModuleList([TemporalAttention(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.grn(x)
        for layer in self.temporal_attention:
            x = layer(x)
        return self.fc(x[:, -1, :])

# Fetch data
def fetch_data(symbol, start_date, end_date):
    try:
        logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for symbol.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

# Technical indicators
def add_technical_indicators(data):
    logging.info("Adding technical indicators")
    df = data.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = compute_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    return df.dropna()

# Compute RSI
def compute_rsi(series, period=14):
    logging.info("Computing RSI")
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Bollinger Bands
def compute_bollinger_bands(series, period=20):
    logging.info("Computing Bollinger Bands")
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (2 * std), sma - (2 * std)

# Preprocess data
def preprocess_data(data):
    logging.info("Preprocessing data")
    data = add_technical_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']
    data = data[features].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create sequences
def create_sequences(data, seq_length):
    logging.info(f"Creating sequences with length {seq_length}")
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length, 3])  # Predict Close price
    return np.array(xs), np.array(ys)

# Train model
def train_model(model, train_loader, epochs, learning_rate, device):
    logging.info("Training model")
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Evaluate model
def evaluate_model(model, test_loader, device):
    logging.info("Evaluating model")
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data).cpu().numpy()
            predictions.extend(outputs.flatten())
            actuals.extend(batch_targets.cpu().numpy().flatten())

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    logging.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
