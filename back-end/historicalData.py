import yfinance as yf

symbol = "^BSESN"

try:
    data = yf.download(symbol, start="2020-01-01", end="2024-12-20")

    if data.empty:
        print(f"No data found for symbol: {symbol}")
    else:
        # Print the data
        print(data)
except Exception as e:
    print(f"Error fetching data for symbol {symbol}: {e}")
