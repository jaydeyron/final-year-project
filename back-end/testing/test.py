import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
import math
import os
import time
import csv

def fetch_options_data(ticker_symbol):
    # Initialize the Ticker object
    stock = yf.Ticker(ticker_symbol)

    # Retrieve available expiration dates
    expirations = stock.options
    print("Available Expiration Dates:", expirations)

    if not expirations:
        print("No options data available.")
        return

    # Fetch the option chain for the first expiration date
    expiration_date = expirations[0]
    print(f"\nFetching options data for expiration date: {expiration_date}")
    option_chain = stock.option_chain(expiration_date)

    # Print calls and puts
    print("\nCalls:")
    print(option_chain.calls.head())  # Print the first few rows of call options

    print("\nPuts:")
    print(option_chain.puts.head())  # Print the first few rows of put options

# Example usage
if __name__ == "__main__":
    ticker = input("Enter the stock name: ") # Example: Apple Inc.
    fetch_options_data(ticker)
