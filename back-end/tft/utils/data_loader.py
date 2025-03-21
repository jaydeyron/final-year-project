import pandas as pd
import numpy as np
import json
import os
import torch
import yfinance as yf
from datetime import datetime, timedelta
import warnings

def fetch_data(symbol, start_date=None, end_date=None, days=None):
    """
    Fetch stock data for the given symbol and date range
    Added 'days' parameter to fetch a specific number of days from end_date
    """
    warnings.filterwarnings('ignore')  # Suppress yfinance warnings
    
    try:
        # If days parameter is provided, calculate start_date based on it
        if days and not start_date:
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            start_date_obj = end_date_obj - timedelta(days=days)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            
        # Download data from Yahoo Finance
        if start_date:
            data = yf.download(symbol, start=start_date, end=end_date)
        else:
            data = yf.download(symbol, end=end_date)
            
        # Check if data is empty
        if data.empty:
            print(f"No data found for {symbol}")
            return None
            
        # Add technical indicators (EMA10, EMA30, RSI, MACD)
        data = add_technical_indicators(data)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def add_technical_indicators(data):
    """Add technical indicators to the dataframe"""
    # Make sure we have enough data
    if len(data) < 30:
        return data
    
    try:
        # Exponential Moving Averages
        data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Forward-fill NaN values created by indicators
        data = data.fillna(method='ffill')
        
        # Backward-fill any remaining NaN values at the start
        data = data.fillna(method='bfill')
        
        return data
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return data

def preprocess_data(data, scaler_params=None):
    """
    Preprocess data: select features, scale features
    
    Parameters:
    - data: DataFrame containing stock data with technical indicators
    - scaler_params: Optional dict with 'mean_' and 'scale_' for normalization
    
    Returns: 
    - scaled_features: numpy array of normalized feature values
    - scaler_params: dict with mean and scale values used for normalization
    """
    # Select relevant features
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA10', 'EMA30', 'RSI', 'MACD', 'MACD_Signal']].values
    
    if scaler_params is None:
        # Calculate mean and standard deviation for scaling
        mean_values = np.mean(features, axis=0)
        std_values = np.std(features, axis=0)
        
        # Create scaler parameters dictionary
        scaler_params = {
            'mean_': mean_values,
            'scale_': std_values
        }
    
    # Scale the features
    scaled_features = (features - scaler_params['mean_']) / scaler_params['scale_']
    
    # Check for NaN values after scaling
    if np.isnan(scaled_features).any():
        # Replace NaN values with 0
        scaled_features = np.nan_to_num(scaled_features, nan=0.0)
    
    return scaled_features, scaler_params

def create_sequences(data, seq_length):
    """
    Create sequences for time series prediction
    Returns: input sequences (X) and target values (y)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Input sequence is of length seq_length
        X.append(data[i:i+seq_length])
        # Target is the next Close price (index 3 is Close price)
        y.append(data[i+seq_length, 3])  # Assuming Close price is at index 3
    
    return np.array(X), np.array(y)

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset class for time series data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for output
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]