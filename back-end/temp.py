import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from tabulate import tabulate
import os

def clear_screen():
    """Clear the terminal screen based on OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_change(value, include_sign=True):
    """Format change values with color codes and signs"""
    if value > 0:
        sign = "+" if include_sign else ""
        return f"\033[92m{sign}{value:.2f}\033[0m"  # Green
    elif value < 0:
        return f"\033[91m{value:.2f}\033[0m"  # Red
    else:
        return f"{value:.2f}"

def get_sensex_data(interval="1d", period="5d"):
    """Get SENSEX index data from Yahoo Finance"""
    try:
        # SENSEX ticker symbol for Yahoo Finance
        ticker = "^BSESN"
        
        # Fetch data
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        
        if data.empty:
            return None
            
        return data
    except Exception as e:
        print(f"Error fetching SENSEX data: {e}")
        return None

def display_current_sensex():
    """Display the current SENSEX index value and daily change"""
    data = get_sensex_data()
    
    if data is None or data.empty:
        print("Could not retrieve SENSEX data")
        return
    
    # Get the latest data point
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else None
    
    # Check if 'latest' contains Series objects and convert to float if needed
    close_value = float(latest['Close']) if isinstance(latest['Close'], pd.Series) else latest['Close']
    open_value = float(latest['Open']) if isinstance(latest['Open'], pd.Series) else latest['Open']
    high_value = float(latest['High']) if isinstance(latest['High'], pd.Series) else latest['High']
    low_value = float(latest['Low']) if isinstance(latest['Low'], pd.Series) else latest['Low']
    
    # Calculate changes
    day_change = close_value - open_value
    day_change_percent = (day_change / open_value) * 100
    
    prev_close_change = 0
    prev_close_change_percent = 0
    prev_close_value = 0
    
    if previous is not None:
        prev_close_value = float(previous['Close']) if isinstance(previous['Close'], pd.Series) else previous['Close']
        prev_close_change = close_value - prev_close_value
        prev_close_change_percent = (prev_close_change / prev_close_value) * 100
    
    # Display current value and daily change
    clear_screen()
    print("\n===== SENSEX CURRENT DATA =====")
    print(f"Date: {latest.name.strftime('%Y-%m-%d')}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nLatest Close: ₹{close_value:.2f}")
    print(f"Day Open:     ₹{open_value:.2f}")
    print(f"Day High:     ₹{high_value:.2f}")
    print(f"Day Low:      ₹{low_value:.2f}")
    
    print(f"\nDay Change:      {format_change(day_change)} ({format_change(day_change_percent)}%)")
    
    if previous is not None:
        print(f"Previous Close:  ₹{prev_close_value:.2f}")
        print(f"From Prev Close: {format_change(prev_close_change)} ({format_change(prev_close_change_percent)}%)")

def display_sensex_history(days=5):
    """Display SENSEX price history for the last several days"""
    # Get data for the specified period
    data = get_sensex_data(period=f"{days}d")
    
    if data is None or data.empty:
        print("Could not retrieve SENSEX history")
        return
    
    # Calculate daily changes
    data['Day Change'] = data['Close'] - data['Open']
    data['Day Change %'] = (data['Day Change'] / data['Open']) * 100
    data['Prev Close Change'] = data['Close'].diff()
    data['Prev Close Change %'] = (data['Prev Close Change'] / data['Close'].shift(1)) * 100
    
    # Create a table with the daily data
    table_data = []
    for idx, row in data.iterrows():
        date = idx.strftime('%Y-%m-%d')
        
        # Convert values to float if they're Series (to avoid formatting errors)
        open_val = float(row['Open']) if isinstance(row['Open'], pd.Series) else row['Open']
        high_val = float(row['High']) if isinstance(row['High'], pd.Series) else row['High']
        low_val = float(row['Low']) if isinstance(row['Low'], pd.Series) else row['Low']
        close_val = float(row['Close']) if isinstance(row['Close'], pd.Series) else row['Close']
        day_change_val = float(row['Day Change']) if isinstance(row['Day Change'], pd.Series) else row['Day Change']
        day_change_pct_val = float(row['Day Change %']) if isinstance(row['Day Change %'], pd.Series) else row['Day Change %']
        
        day_change = format_change(day_change_val)
        day_change_pct = format_change(day_change_pct_val)
        
        # Handle the Prev Close values which might be NaN for the first row
        if pd.isna(row['Prev Close Change']):
            prev_close_change = "N/A"
            prev_close_pct = "N/A"
        else:
            prev_change_val = float(row['Prev Close Change']) if isinstance(row['Prev Close Change'], pd.Series) else row['Prev Close Change']
            prev_pct_val = float(row['Prev Close Change %']) if isinstance(row['Prev Close Change %'], pd.Series) else row['Prev Close Change %']
            prev_close_change = format_change(prev_change_val)
            prev_close_pct = format_change(prev_pct_val)
        
        table_data.append([
            date, 
            f"₹{open_val:.2f}", 
            f"₹{high_val:.2f}", 
            f"₹{low_val:.2f}", 
            f"₹{close_val:.2f}",
            f"{day_change} ({day_change_pct}%)",
            f"{prev_close_change} ({prev_close_pct}%)"
        ])
    
    # Format and display the table
    headers = ["Date", "Open", "High", "Low", "Close", "Day Change", "From Prev Close"]
    print("\n===== SENSEX HISTORICAL DATA =====")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

def one_time_display():
    """Display SENSEX data just once without continuous monitoring"""
    try:
        # Clear the screen for a clean display
        clear_screen()
        
        # Show current SENSEX data
        display_current_sensex()
        
        # Show historical data
        display_sensex_history(days=10)  # Show 10 days of history instead of default 5
        
        print("\nData displayed successfully. Run the script again for updated data.")
        
    except Exception as e:
        print(f"Error displaying data: {e}")

if __name__ == "__main__":
    # Check if tabulate is installed, if not provide instructions
    try:
        import tabulate
    except ImportError:
        print("The 'tabulate' package is required. Please install it with:")
        print("pip install tabulate")
        exit(1)
        
    # Display the SENSEX data once instead of continuous monitoring
    one_time_display()
