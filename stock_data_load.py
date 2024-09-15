import yfinance as yf
import psycopg2
import datetime
import pandas as pd

# Database connection setup
def get_db_connection():
    return psycopg2.connect(
        host="localhost",  # Replace with your PostgreSQL host
        database="finai",  # Replace with your database name
        user="postgres",  # Replace with your username
        password="superduper1!A"  # Replace with your password
    )

# Function to insert stock data into PostgreSQL
def insert_stock_data(stock_data, symbol, conn):
    with conn.cursor() as cur:
        for date, row in stock_data.iterrows():
            query = """
            INSERT INTO stock_prices (symbol, date, open, high, low, close)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date)
            DO NOTHING;
            """
            
            data = (symbol, date, row['Open'], row['High'], row['Low'], row['Close'])
            cur.execute(query, data)
        conn.commit()
        print(f"Data for {symbol} inserted successfully.")

# Function to download stock data and insert directly into the database
def download_and_insert_stock_data(stock_symbols, conn, start, end ):
    for symbol in stock_symbols:
        try:
            print(f"Downloading data for {symbol}...")
            stock_df = yf.download(symbol, start=start, end=end)
            if not stock_df.empty:
                # Insert data into PostgreSQL
                insert_stock_data(stock_df[['Open', 'High', 'Low', 'Close']], symbol, conn)
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Failed to download or insert data for {symbol}: {e}")

# Function to get stock symbols from a CSV file
def get_sp500_stocks_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['Symbol'].tolist()

if __name__ == "__main__":
    # Define the date range for the past 5 years
    end_date = datetime.datetime.today().date()
    start_date = end_date - datetime.timedelta(days=5*365)

    # Database connection
    conn = get_db_connection()

    try:
        # Read S&P 500 stock symbols from CSV file
        sp500_symbols = get_sp500_stocks_from_csv('stock_data/metadata.csv')

        # Download and insert stock data directly into the PostgreSQL table
        download_and_insert_stock_data(sp500_symbols,conn, start=start_date, end=end_date)
    finally:
        # Close the database connection
        conn.close()

    print("All data downloaded and inserted successfully.")
