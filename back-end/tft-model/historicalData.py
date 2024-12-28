import yfinance as yf

# Define the stock symbol
symbol = "^NSEI"  # Example stock symbol

# Fetch historical data
data = yf.download(symbol, start="2020-01-01", end="2024-12-20")

# Print the data
print(data)
