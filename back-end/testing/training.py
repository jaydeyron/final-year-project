import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Fetch historical Nifty data
def fetch_nifty_data(start_date='2010-01-01', end_date='2024-01-01'):
    nifty = yf.Ticker("^NSEI")  # ^NSEI is the ticker symbol for Nifty 50
    data = nifty.history(start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.reset_index(inplace=True)  # Reset index to get 'Date' as a column
    return data

# Data Preparation
def load_and_preprocess_data(seq_length):
    data = fetch_nifty_data()
    
    # Prepare features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    target = data['Close'].shift(-1).dropna().values  # Predict next day's Close
    
    # Drop the last row since its target value will be NaN
    features = features[:len(target)]
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Create sequences
    def create_sequences(data, target, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            tar = target[i+seq_length]
            sequences.append(seq)
            targets.append(tar)
        return np.array(sequences), np.array(targets)
    
    X, y = create_sequences(scaled_features, target, seq_length)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

# Model Definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(model_dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])  # Output the prediction for the last time step
        return x

# Training Function
def train_model(model, X_train, y_train, epochs=20, batch_size=32, lr=1e-5):
    criterion = nn.MSELoss()  # Use Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
            targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluation Function
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        targets = torch.tensor(y_test, dtype=torch.float32)
        predictions = model(inputs).squeeze().numpy()
        
    mse = np.mean((predictions - y_test) ** 2)
    print(f'Mean Squared Error on Test Set: {mse}')

# Main Script
if __name__ == "__main__":
    # Parameters
    seq_length = 30  # Number of days in each sequence
    input_dim = 5  # Number of features
    model_dim = 64
    num_heads = 8
    num_layers = 3
    dropout = 0.2
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(seq_length)
    
    # Initialize and train model
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, dropout)
    train_model(model, X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
