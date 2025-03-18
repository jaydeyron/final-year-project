import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import ta
import torch
from torch.utils.data import Dataset
from config import Config  # Add this import

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
    """Fetch stock data and add technical indicators"""
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Add technical indicators
    df['MA_10'] = ta.trend.sma_indicator(df.Close, window=10)
    df['RSI'] = ta.momentum.rsi(df.Close)
    macd = ta.trend.MACD(df.Close)
    df['MACD'] = macd.macd()
    
    bb = ta.volatility.BollingerBands(df.Close)
    df['Upper_Band'] = bb.bollinger_hband()
    df['Lower_Band'] = bb.bollinger_lband()
    
    # Forward fill NaN values
    df = df.fillna(method='ffill')
    
    return df

def preprocess_data(data, scaler=None):
    """Preprocess data and apply scaling"""
    features = data[Config.FEATURES].values
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    return features, scaler

def create_sequences(data, seq_length):
    """Create sequences for training"""
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 3]  # Using Close price as target
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)