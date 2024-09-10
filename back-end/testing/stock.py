import yfinance as yf # type: ignore

# Define the ticker symbol for Nifty 50 Index
ticker = "^NSEI"  # Nifty 50 Index symbol on Yahoo Finance

# Fetch the ticker data
index = yf.Ticker(ticker)

# Fetch options expiration dates
try:
    expiration_dates = index.options
    print("Options Expiration Dates:")
    print(expiration_dates)

    # Fetch and display options data for the first expiration date
    if expiration_dates:
        expiry_date = expiration_dates[0]
        options_data = index.option_chain(expiry_date)
        
        print(f"\nOptions Data for Expiration Date: {expiry_date}")
        print("\nCalls:")
        print(options_data.calls.head())  # Display the first few call options

        print("\nPuts:")
        print(options_data.puts.head())  # Display the first few put options
    else:
        print("No options data available.")
except Exception as e:
    print(f"An error occurred: {e}")
