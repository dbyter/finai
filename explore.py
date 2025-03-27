from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


DATA_FOLDER = "data/"
# List of stock symbols to download
tickers = ["GOOG", "AAPL", "NVDA"]
all_data = {}
for ticker in tickers: 
    file_name = f"{DATA_FOLDER}{ticker}_10yr_data.csv"
    # Read file with pandas 
    data = pd.read_csv(file_name)
    # Convert date to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    data["TICKER"] = ticker
    # Set date as index
    data.set_index("Date", inplace=True)
    # Calculate daily returns
    data["daily_return"] = data["Close"].pct_change()
    # Calculate daily log returns
    data["log_return"] = np.log(1 + data["daily_return"])
    # Calculate daily volatility
    data["daily_volatility"] = data["log_return"].rolling(window=252).std()
    # Calculate annual volatility
    data["annual_volatility"] = data["daily_volatility"] * np.sqrt(252)
    # Provide some statistics 
    # Average daily return: 
    avg_daily_return = data["daily_return"].mean()
    print(f"Average daily return for {ticker}: {avg_daily_return}")
    # Average daily volatility:
    avg_daily_volatility = data["daily_volatility"].mean()
    print(f"Average daily volatility for {ticker}: {avg_daily_volatility}")

    # Total return by year 
    total_return = data.groupby(data.index.year)["daily_return"].sum()
    print(f"Total return by year for {ticker}:")
    print(total_return)

    # Total return by year if you only held the stock on days where return was positive
    data["positive_return"] = data.apply(lambda x: x["daily_return"] if x["daily_return"] > 0.02 else 0, axis=1)
    total_return_positive = data.groupby(data.index.year)["positive_return"].sum()
    print(f"Total return by year for {ticker} if you only held the stock on days where return was positive:")
    print(total_return_positive)
    # Print 5 rows of data to check
    all_data[ticker] = data

