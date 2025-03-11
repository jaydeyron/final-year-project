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

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Temporal Fusion Transformer model
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# Data fetching
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data found for symbol.")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

# Adding more technical indicators
def add_technical_indicators(data):
    df = data.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = compute_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    return df.dropna()

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Bollinger Bands
def compute_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (2 * std), sma - (2 * std)

# Data preprocessing
def preprocess_data(data):
    data = add_technical_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']
    data = data[features].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler.pkl')
    return scaled_data, scaler

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length, 3])
    return np.array(xs), np.array(ys)

# Training function
def train_model(model, train_loader, epochs, learning_rate, device):
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nStarting training for {epochs} epochs...")
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), 'tft_model.pth')


def main():
    symbol = "^BSESN"
    start_date = "1986-01-01"
    end_date = "2025-02-11"
    seq_length = 60
    hidden_size = 128
    num_heads = 4
    num_layers = 4
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Fetching data...")
    data = fetch_data(symbol, start_date, end_date)

    print("Preprocessing data...")
    scaled_data, scaler = preprocess_data(data)

    print("Creating sequences...")
    xs, ys = create_sequences(scaled_data, seq_length)

    print("Creating dataloader...")
    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = xs.shape[2]
    model = TemporalFusionTransformer(input_size, hidden_size, num_heads, num_layers, 1)

    train_model(model, dataloader, epochs, learning_rate, device)

if __name__ == "__main__":
    main()
