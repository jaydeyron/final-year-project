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
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score, accuracy_score
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
        all_predictions, all_targets = [], []
        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().detach().numpy().flatten())
            all_targets.extend(batch_targets.cpu().numpy().flatten())

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

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
    r2 = r2_score(actuals, predictions)
    logging.info(f"Final Evaluation -> MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function
def main():
    symbol = "^BSESN"
    start_date = "2020-01-01"
    end_date = "2025-02-27"
    seq_length = 60
    hidden_size = 128
    num_layers = 4
    batch_size = 64
    epochs = 100
    learning_rate = 0.0005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Starting main function")
    data = fetch_data(symbol, start_date, end_date)
    scaled_data, scaler = preprocess_data(data)
    xs, ys = create_sequences(scaled_data, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, shuffle=False)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TemporalFusionTransformer(xs.shape[2], hidden_size, 1, num_layers)
    train_model(model, train_loader, epochs, learning_rate, device)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
