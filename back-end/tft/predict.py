import torch
from utils.data_loader import fetch_data, preprocess_data, create_sequences
from config import Config
from models.tft_model import TemporalFusionTransformer

def main():
    # Load model
    input_size = len(Config.FEATURES)  # Use the FEATURES attribute from Config
    model = TemporalFusionTransformer(input_size, Config.HIDDEN_SIZE, 1, Config.NUM_LAYERS, Config.NUM_HEADS, Config.DROPOUT)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.to(Config.DEVICE)
    model.eval()

    # Fetch latest data
    data = fetch_data("^BSESN", "2010-01-01", "2025-03-18")  # Example: Fetch latest data
    scaled_data, scaler = preprocess_data(data)

    # Create sequence for prediction
    last_sequence = torch.tensor(scaled_data[-Config.SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

    # Make prediction
    with torch.no_grad():
        prediction = model(last_sequence).cpu().numpy()[0, 0]
        predicted_price = scaler.inverse_transform([[0, 0, 0, prediction, 0, 0, 0, 0, 0, 0]])[0, 3]

    print(f"Predicted next day's closing price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()