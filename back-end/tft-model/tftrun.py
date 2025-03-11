import torch
import joblib
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# 1. Define the model architecture (must match exactly with training code)
class GatedResidualNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_residual = torch.nn.Linear(input_dim, output_dim)
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.gate = torch.nn.Linear(output_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.elu(x1)
        x1 = self.fc2(x1)
        x1 = self.dropout(x1)
        gate = self.sigmoid(self.gate(x1))
        x_residual = self.fc_residual(x)
        return gate * x1 + (1 - gate) * x_residual

class TemporalFusionTransformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.variable_selection = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.variable_selection(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# 2. Helper functions for data processing
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (2 * std), sma - (2 * std)

def add_technical_indicators(data):
    df = data.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = compute_rsi(df['Close'])
    df['BB_upper'], df['BB_lower'] = compute_bollinger_bands(df['Close'])
    return df.dropna()

# 3. Prediction function
def predict_next_close():
    # Load saved artifacts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load('scaler.pkl')
    
    # Model parameters
    input_size = 10
    hidden_size = 128
    num_heads = 4
    num_layers = 4
    seq_length = 60
    
    # Initialize model
    model = TemporalFusionTransformer(input_size, hidden_size, num_heads, num_layers, 1)
    model.load_state_dict(torch.load('tft_model.pth', map_location=device))
    model.eval()
    model.to(device)
    
    # Fetch data
    data = yf.download("^BSESN", period="60d")
    
    # Preprocess
    processed_data = add_technical_indicators(data)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']
    processed_data = processed_data[features].dropna()
    
    # Scale data
    scaled_data = scaler.transform(processed_data[-seq_length:])
    
    # Prepare input
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0][0]
    
    # Inverse transform
    dummy = np.zeros((1, len(features)))
    dummy[0, 3] = prediction
    predicted_price = scaler.inverse_transform(dummy)[0, 3]
    
    # Get current price (FIXED)
    current_price = data['Close'].iloc[-1].item()  # Convert to scalar
    
    print(f"\nCurrent Price: {current_price:.2f}")
    print(f"Predicted Next Close: {predicted_price:.2f}")

if __name__ == "__main__":
    predict_next_close()