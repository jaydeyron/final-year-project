import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

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

# Model: Improved LSTM-based TFT
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

def main():
    symbol = "^BSESN"
    start_date = "2020-01-01"
    end_date = "2025-02-11"
    seq_length = 60
    hidden_size = 128
    num_layers = 4
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Fetching data...")
    data = fetch_data(symbol, start_date, end_date)
    current_price = float(data['Close'].iloc[-1])

    print("Preprocessing data...")
    scaled_data, scaler = preprocess_data(data)

    print("Creating sequences...")
    xs, ys = create_sequences(scaled_data, seq_length)

    print("Creating dataloader...")
    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nInitializing model with:")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of layers: {num_layers}")
    print(f"Sequence length: {seq_length}")
    print(f"Using device: {device}")

    input_size = xs.shape[2]
    model = TemporalFusionTransformer(input_size, hidden_size, 1, num_layers)

    train_model(model, dataloader, epochs, learning_rate, device)

    print("\nMaking prediction...")
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(scaled_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(last_sequence).cpu().numpy()[0, 0]
        predicted_price = scaler.inverse_transform([[0, 0, 0, prediction, 0, 0, 0, 0, 0, 0]])[0, 3]

        print(f"\nCurrent price: {current_price:.2f}")
        print(f"Predicted next day's closing price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()
