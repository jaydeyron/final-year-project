import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time-series data.
    """
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def fetch_data(symbol, start_date, end_date):
    """
    Fetch stock data using yfinance.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found for symbol.")
    return data

def add_technical_indicators(data):
    """
    Add technical indicators to the stock data.
    """
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    df['RSI_14'] = compute_rsi(df['Close'])
    
    # Bollinger Bands
    df['BB_upper'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    
    return df.dropna()

def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20):
    """
    Compute Bollinger Bands.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (2 * std), sma - (2 * std)

def preprocess_data(data):
    """
    Preprocess the stock data by adding technical indicators and scaling.
    """
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']
    data = data[features].dropna()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data, seq_length):
    """
    Create sequences of data for time-series forecasting.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length, 3])  # Target is the 'Close' price
    return np.array(xs), np.array(ys)