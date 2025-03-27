import yfinance as yf
import pandas as pd
import datetime


# Function to get the list of S&P 500 stocks (using an online source)
def get_sp500_stocks_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['Symbol'].tolist()


# Define the date range for the past 5 years
end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(days=10*365)

# Function to download historical data for each stock
def download_stock_data(stock_symbols, start, end):
    stock_data = {}
    
    for symbol in stock_symbols:
        try:
            print(f"Downloading data for {symbol}...")
            stock_df = yf.download(symbol, start=start, end=end)
            stock_df.reset_index(inplace=True)
            stock_df.columns = stock_df.columns.droplevel('Ticker')
            if not stock_df.empty:
                stock_data[symbol] = stock_df[['Date','Open', 'Close', 'Low', 'High', 'Volume']]
        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")
    
    return stock_data

# Save each stock's data as a CSV
def save_to_csv(stock_data):
    for symbol, data in stock_data.items():
        file_name = f"data/{symbol}_10yr_data.csv"
        data.to_csv(file_name)
        print(f"Saved {symbol} data to {file_name}")

if __name__ == "__main__":
    # Get S&P 500 stock symbols
    # sp500_symbols = get_sp500_stocks_from_csv('metadata.csv')
    
    # Download the historical stock data
    stock_data = download_stock_data(['GOOG', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA',
    'UNH', 'XOM', 'LLY', 'JPM', 'V', 'PG', 'AVGO', 'MRK', 'COST', 'PEP', 'ADBE', 
    'HUM', 'ELV', 'CVS', 'CNC', 'CI',
    'ALL', 'AXP', 'AIG', 'AMGN', 'AON', 'T', 'BKR', 'BBY', 'BA', 'COF', 'CSCO'
    ], start=start_date, end=end_date)
    
    # Save the data to CSV files
    save_to_csv(stock_data)

    print("Download complete.")
