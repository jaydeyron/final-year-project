import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import ta
import torch
from torch.utils.data import Dataset
from config import Config

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

def fetch_data(symbol, start_date=None, end_date=None, days=None):
    """Fetch stock data and add technical indicators"""
    try:
        # If days is provided, use it for relative date range
        if days is not None and start_date is None:
            from datetime import datetime, timedelta
            end_date = end_date or datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Download the data
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol} in the specified date range")
        
        # Make sure Close column is a Series, not a DataFrame
        close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].squeeze()
        
        # Add technical indicators - ensuring we're passing Series, not DataFrames
        df['MA_10'] = ta.trend.sma_indicator(close=close_series, window=10)
        df['RSI'] = ta.momentum.rsi(close=close_series)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(close=close_series)
        df['MACD'] = macd.macd()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=close_series)
        df['Upper_Band'] = bb.bollinger_hband()
        df['Lower_Band'] = bb.bollinger_lband()
        
        # Forward fill NaN values that occur at the beginning of indicators
        df = df.fillna(method='ffill')
        # Back fill any remaining NaN values
        df = df.fillna(method='bfill')
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        raise e

def preprocess_data(data, scaler_params=None):
    """Preprocess data and apply scaling"""
    # Make sure we have all required features
    required_features = Config.FEATURES
    
    # Check if all required features exist in the data
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Extract selected features
    features = data[required_features].values
    
    if scaler_params is None:
        # Create and fit new scaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
        return scaled_data, scaler
    else:
        # Create scaler with provided parameters
        scaler = StandardScaler()
        
        # Set the mean_ and scale_ directly
        scaler.mean_ = np.array(scaler_params['mean_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        
        # Transform the data
        scaled_data = scaler.transform(features)
        return scaled_data, scaler

def create_sequences(data, seq_length):
    """Create sequences for training"""
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 3]  # Using Close price as target
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)