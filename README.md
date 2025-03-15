# Stock Market Prediction using Temporal Fusion Transformer

## Project Overview
This project implements a sophisticated stock market prediction system using the Temporal Fusion Transformer (TFT) architecture. The system analyzes historical stock data and various technical indicators to forecast future stock prices, providing traders and investors with data-driven insights.

## Technologies Used
- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework for TFT implementation
- **FastAPI**: Backend API development
- **React**: Frontend user interface
- **yfinance**: Yahoo Finance data retrieval
- **pandas & numpy**: Data manipulation and analysis
- **scikit-learn**: Data preprocessing and evaluation

## Model Architecture
The system utilizes a Temporal Fusion Transformer (TFT) with:
1. **Variable Selection Network**: For dynamic feature selection
2. **Gated Residual Network**: For processing static and temporal features
3. **Multi-Head Attention**: For capturing temporal dependencies
4. **LSTM layers**: For sequential data processing

## Features
- Real-time stock data fetching
- Technical indicator integration:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- Interactive visualization dashboard
- Multi-step forecasting
- Model interpretation capabilities

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start backend: `uvicorn main:app --reload`
4. Start frontend: `npm start`

## API Endpoints
- `/predict`: Get stock price predictions
- `/historical`: Fetch historical data
- `/indicators`: Get technical indicators

## Performance Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

## Future Enhancements
- Integration of sentiment analysis
- Support for multiple stock symbols
- Advanced feature engineering
- Portfolio optimization suggestions
