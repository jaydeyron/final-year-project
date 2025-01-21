import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.3):
        super(TemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()  
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create sequences for the model
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 3]  # Predict the 'Close' price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Example usage
if __name__ == "__main__":
    # Parameters
    symbol = "^BSESN"
    start_date = "1986-01-01" 
    end_date = "2025-01-17"  
    seq_length = 30
    input_size = 5  # Number of features
    hidden_size = 100  # Increased hidden size
    output_size = 1
    num_layers = 4  # Increased number of layers
    batch_size = 64  # Increased batch size
    epochs = 100  # Increased number of epochs
    learning_rate = 0.001

    data = fetch_data(symbol, start_date, end_date)
    current_price = data['Close'].iloc[-1]
    data, scaler = preprocess_data(data)
    data_values = data

    xs, ys = create_sequences(data_values, seq_length)

    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)  # Ensure target shape matches output shape

    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalFusionTransformer(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()  # Adjust learning rate
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    torch.save(model.state_dict(), 'tft_model.pth')

    # Predict the next day's price
    model.eval()
    with torch.no_grad():
        last_sequence = torch.tensor(data_values[-seq_length:], dtype=torch.float32).unsqueeze(0)
        next_day_prediction = model(last_sequence).item()
        next_day_prediction = scaler.inverse_transform([[0, 0, 0, next_day_prediction, 0]])[0, 3]  # Inverse transform the 'Close' price
        print(f"Current price: {current_price}")
        print(f"Predicted next day's closing price: {next_day_prediction}")