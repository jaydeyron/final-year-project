import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data[['Close']]
    data = data.dropna()  
    scaler = MinMaxScaler()
    data['Close'] = scaler.fit_transform(data[['Close']])
    return data, scaler

# Create sequences for the model
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Example usage
if __name__ == "__main__":
    # Parameters
    symbol = "^BSESN"
    start_date = "2015-01-01" 
    end_date = "2019-12-31"  
    seq_length = 30
    input_size = 1
    hidden_size = 50  
    output_size = 1
    num_layers = 3  
    batch_size = 32
    epochs = 50 
    learning_rate = 0.001


    data = fetch_data(symbol, start_date, end_date)
    data, scaler = preprocess_data(data)
    data_values = data.values


    xs, ys = create_sequences(data_values, seq_length)


    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.float32)


    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = TemporalFusionTransformer(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")


    torch.save(model.state_dict(), 'tft_model.pth')