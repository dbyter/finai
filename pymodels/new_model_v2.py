import pandas as pd 
import numpy as np 
import logging
from pymongo.mongo_client import MongoClient
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataModel: 
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.data_folder = DATA_FOLDER
        self.file_name = "data_with_fundamentals.pkl"
        # Create data folder if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)

    def get_data (self):
        if self.use_cache and os.path.exists(f"{self.data_folder}/{self.file_name}"):
            return self.load_from_cache()
        else:
            logger.info("Loading data from MongoDB")
            data = self.load_data_from_mongo()
            logger.info("Saving data to pickle file")
            self.save_data_frame(data, self.file_name)
            return data

    def load_from_cache(self):
        logger.info("Loading data from cache")
        with open(f"{self.data_folder}/{self.file_name}", 'rb') as f:
            return pickle.load(f)

    def save_data_frame (self, data_dict, filename):
        logger.info(f"Saving data for {len(data_dict)} tickers")
        with open(f"{self.data_folder}/{filename}", 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"Saved data to {self.data_folder}/{filename}")

    def process_ticker_data(self, ticker_data):
        ticker = ticker_data["ticker"]
        stock_data = ticker_data["data"]
        df = pd.DataFrame(stock_data)
        df["Date"] = pd.to_datetime(df["date"])
        df.set_index("Date", inplace=True)
        df.drop(columns=["date"], inplace=True)
        df.rename(columns={"adjusted_close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)
        df = df.sort_index( ascending = True )

        # Calculate various returns 
        df['Return'] = (df['Close'] - df['Open'] ) / df['Open']
        df['Price_Diff'] = df['Close'] - df['Open']
        df['Next_Day_Return'] = df['Return'].shift(-1)
        df['Next_7_Day_Return'] = df['Return'].rolling(window=7).sum().shift(-6)
        df['Next_7_Day_Return_StdDev'] = df['Return'].rolling(window=7).std().shift(-6)
        df['Lagged_Return_1'] = df['Return'].shift(1)
        df['Lagged_Return_2'] = df['Return'].shift(2)
        df['Lagged_Return_3'] = df['Return'].shift(3)
        df['Lagged_Return_4'] = df['Return'].shift(4)
        df['Lagged_Return_5'] = df['Return'].shift(5)
        df['Lagged_Return_6'] = df['Return'].shift(6)
        df['Lagged_Return_7'] = df['Return'].shift(7)

        # Remove outliers 
        df = df[(np.abs(df['Return']) < 0.1)]

        # Create classification labels for 7 day return 
        df['Highly_Positive_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if x > 0.02 else 0)
        df['Highly_Negative_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if x < -0.02 else 0)
        df['Positive_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if 0.02 >= x > 0 else 0)
        df['Negative_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if 0 > x >= -0.02 else 0)

        # Add month and day of the week as features
        df['Month'] = df.index.month
        df['Is_January'] = df['Month'].apply(lambda x: 1 if x == 1 else 0)
        df['Is_February'] = df['Month'].apply(lambda x: 1 if x == 2 else 0)
        df['Is_March'] = df['Month'].apply(lambda x: 1 if x == 3 else 0)
        df['Is_April'] = df['Month'].apply(lambda x: 1 if x == 4 else 0)
        df['Is_May'] = df['Month'].apply(lambda x: 1 if x == 5 else 0)
        df['Is_June'] = df['Month'].apply(lambda x: 1 if x == 6 else 0)
        df['Is_July'] = df['Month'].apply(lambda x: 1 if x == 7 else 0)
        df['Is_August'] = df['Month'].apply(lambda x: 1 if x == 8 else 0)
        df['Is_September'] = df['Month'].apply(lambda x: 1 if x == 9 else 0)
        df['Is_October'] = df['Month'].apply(lambda x: 1 if x == 10 else 0)
        df['Is_November'] = df['Month'].apply(lambda x: 1 if x == 11 else 0)
        df['Is_December'] = df['Month'].apply(lambda x: 1 if x == 12 else 0)
        df['Day_of_week'] = df.index.dayofweek
        df['Is_Monday'] = df['Day_of_week'].apply(lambda x: 1 if x == 0 else 0)
        df['Is_Tuesday'] = df['Day_of_week'].apply(lambda x: 1 if x == 1 else 0)
        df['Is_Wednesday'] = df['Day_of_week'].apply(lambda x: 1 if x == 2 else 0)
        df['Is_Thursday'] = df['Day_of_week'].apply(lambda x: 1 if x == 3 else 0)
        df['Is_Friday'] = df['Day_of_week'].apply(lambda x: 1 if x == 4 else 0)

        # Add the year as a feature
        df['Year'] = df.index.year
        df['Is_2020'] = df['Year'].apply(lambda x: 1 if x == 2020 else 0)
        df['Is_2019'] = df['Year'].apply(lambda x: 1 if x == 2019 else 0)
        df['Is_2018'] = df['Year'].apply(lambda x: 1 if x == 2018 else 0)
        df['Is_2017'] = df['Year'].apply(lambda x: 1 if x == 2017 else 0)
        df['Is_2016'] = df['Year'].apply(lambda x: 1 if x == 2016 else 0)
        df['Is_2015'] = df['Year'].apply(lambda x: 1 if x == 2015 else 0)
        df['Is_2014'] = df['Year'].apply(lambda x: 1 if x == 2014 else 0)
        df['Is_2021'] = df['Year'].apply(lambda x: 1 if x == 2021 else 0)
        df['Is_2022'] = df['Year'].apply(lambda x: 1 if x == 2022 else 0)
        df['Is_2023'] = df['Year'].apply(lambda x: 1 if x == 2023 else 0)
        df['Is_2024'] = df['Year'].apply(lambda x: 1 if x == 2024 else 0)
        df['Is_2025'] = df['Year'].apply(lambda x: 1 if x == 2025 else 0)


        # Add year
        df['Year'] = df.index.year
        df['30_day_avg'] = df['Close'].rolling(window=30).mean()
        # Add 7 day moving average
        df['7_day_avg'] = df['Close'].rolling(window=7).mean()
        # Add the minimum price in the last 30 days
        df['30_day_min'] = df['Close'].rolling(window=30).min()
        # Add the maximum price in the last 30 days
        df['30_day_max'] = df['Close'].rolling(window=30).max()
        # Add the minimum price in the last 7 days
        df['7_day_min'] = df['Close'].rolling(window=7).min()
        # Add the maximum price in the last 7 days
        df['7_day_max'] = df['Close'].rolling(window=7).max()


        # add similar metrics but based on return
        df['30_day_avg_return'] = df['Return'].rolling(window=30).mean()
        # Add 7 day moving average
        df['7_day_avg_return'] = df['Return'].rolling(window=7).mean()
        # Add the minimum price in the last 30 days
        df['30_day_min_return'] = df['Return'].rolling(window=30).min()
        # Add the maximum price in the last 30 days
        df['30_day_max_return'] = df['Return'].rolling(window=30).max()
        # Cumulative return over 7 14 and 30 days
        df['cumulative_return_7'] = df['Return'].rolling(window=7).sum()
        df['cumulative_return_14'] = df['Return'].rolling(window=14).sum()
        df['cumulative_return_30'] = df['Return'].rolling(window=30).sum()

        # Add RSI
        # Calculate the 14-day RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Add OBV
        df['Volume'] = df['Volume'].astype(float)
        df['OBV'] = (np.sign(df['Return']) * df['Volume']).cumsum()

        # Create categorical features based on the value of return (highly positive, highly negative, positive, negative)
        df['Highly_Positive'] = df['Return'].apply(lambda x: 1 if x > 0.02 else 0)
        df['Highly_Negative'] = df['Return'].apply(lambda x: 1 if x < -0.02 else 0)
        df['Positive'] = df['Return'].apply(lambda x: 1 if 0.02 >= x > 0 else 0)
        df['Negative'] = df['Return'].apply(lambda x: 1 if 0 > x >= -0.02 else 0)

        # Add a random column to test the model
        df['Random'] = np.random.randint(0, 2, size=len(df))
        df.dropna(inplace=True)

        return ticker, df


    def load_data_from_mongo(self):
        def load_fundamentals_data_from_mongo(companies_collection, balance_sheet_collection, income_statement_collection, cash_flow_collection):
            logger.info("Starting data loading for all fundamentals")
            current_time = datetime.now()
            balance_sheet_data = list(balance_sheet_collection.find({"report_type": "10K"}))
            balance_sheet_data = pd.DataFrame(balance_sheet_data)
            balance_sheet_data.set_index("date", inplace=True)
            balance_sheet_data.drop(columns=["_id"], inplace=True)

            logger.info(f"Balance sheet data loaded: {len(balance_sheet_data)}")
            income_statement_data = list(income_statement_collection.find({"report_type": "10K"}))
            income_statement_data = pd.DataFrame(income_statement_data)
            income_statement_data.set_index("date", inplace=True)
            income_statement_data.drop(columns=["_id"], inplace=True)
            logger.info(f"Income statement data loaded: {len(income_statement_data)}")
            cash_flow_data = list(cash_flow_collection.find({"report_type": "10K"}))
            cash_flow_data = pd.DataFrame(cash_flow_data)
            cash_flow_data.set_index("date", inplace=True)
            cash_flow_data.drop(columns=["_id"], inplace=True)
            logger.info(f"Cash flow data loaded: {len(cash_flow_data)}")
            logger.info(f"Fundamentals data loading completed in {datetime.now() - current_time}")

            # Merge balance sheet, income statement, and cash flow data on ticker and date 
            merged_data = pd.merge(balance_sheet_data, income_statement_data, on=["ticker", "filing_date"], how="inner")
            merged_data = pd.merge(merged_data, cash_flow_data, on=["ticker", "filing_date"], how="inner")
            logger.info(f"Merged data fundamentals loaded: {len(merged_data)}")
            return merged_data

        def load_stock_data_from_mongo(stock_collection):
            logger.info("Starting stock data loading")
            logger.info("Starting stock aggregation pipeline")
            # Optimize the query to only fetch required fields
            aggregation_pipeline = [
                {"$project": {
                    "ticker": 1,
                    "date": 1,
                    "adjusted_close": 1,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "volume": 1
                }},
                {"$group": {
                    "_id": "$ticker",
                    "data": {"$push": "$$ROOT"}
                }},
                {"$project": {
                    "_id": 0,
                    "ticker": "$_id",
                    "data": 1
                }}
            ]
            
            mongo_stock_data = list(stock_collection.aggregate(aggregation_pipeline))
            logger.info(f"Stock aggregation pipeline completed in {datetime.now() - current_time}")

            current_time_processing = datetime.now()
            logger.info("Starting data processing")
            
            # Process data in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(mongo_stock_data), 8)) as executor:
                results = list(executor.map(self.process_ticker_data, mongo_stock_data))
            
            # Convert results to dictionary
            all_data = dict(results)
            
            logger.info(f"Data processing completed in {datetime.now() - current_time_processing}")
            logger.info(f"Data loading completed in {datetime.now() - current_time}")
            return all_data

        logger.info("Starting data loading for all tickers")
        current_time = datetime.now()    
        db_params = {
            "uri": "mongodb+srv://ihebselmi:qHaevmEpLV96SDCF@cluster1.ecb47oc.mongodb.net/",
            "db_name": "financial_data",
        }
        
        client = MongoClient(db_params["uri"])
        db = client[db_params["db_name"]]
        stock_collection = db.stock
        balance_sheet_collection = db.balance_sheets
        income_statement_collection = db.income_statements
        cash_flow_collection = db.cash_flows
        companies_collection = db.companies

        stock_data = load_stock_data_from_mongo(stock_collection)
        fundamentals_df = load_fundamentals_data_from_mongo(companies_collection, balance_sheet_collection, income_statement_collection, cash_flow_collection)

        # Merge stock and fundamentals data for each ticker
        merged_data = {}
        for ticker, stock_df in stock_data.items():
            if ticker in fundamentals_df['ticker'].unique():
                # Get fundamentals for this ticker
                ticker_fundamentals = fundamentals_df[fundamentals_df['ticker'] == ticker]
                
                # Merge stock data with fundamentals
                merged_df = pd.merge_asof(
                    stock_df.sort_values('Date'),
                    ticker_fundamentals.sort_values('filing_date'),
                    left_on='Date',
                    right_on='filing_date',
                    by='ticker',
                    direction='backward'
                )
                merged_data[ticker] = merged_df

        logger.info(f"Successfully merged data for {len(merged_data)} tickers")
        logger.info(merged_data['AAPL'].head())
        return merged_data


    
    

# # Preprocessing function
# def preprocess_data(df):
#     logger.debug("Starting data preprocessing")
#     # Add return and month columns

#     # Scale data
#     scaler = StandardScaler()
#     # scaled_data = scaler.fit_transform(df[['Return', 'Month']])
#     # scaled_data = scaler.fit_transform(df[['Return', 'Month', 'Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', '30_day_avg_return', '7_day_avg_return', 
#     # '30_day_min_return', '30_day_max_return', 'RSI', 'Volume', 'OBV']])
#     scaled_data = scaler.fit_transform(
#         df[[
#         'Return','Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', 
#         'Lagged_Return_4', 'Lagged_Return_5', 'Lagged_Return_6', 'Lagged_Return_7',
#         'RSI', 'OBV',
#         'Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 
#         'Is_June', 'Is_July', 'Is_August', 'Is_September', 
#         'Is_October', 'Is_November', 'Is_December',
#         'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday',
#         'Is_2020', 'Is_2019', 'Is_2018', 'Is_2017', 'Is_2016', 'Is_2015', 'Is_2014', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024',
#         'Random'
#         ]]
#     )
#     scaled_data_Y = np.array(df[['Next_Day_Return', 'Next_7_Day_Return_StdDev']])

#     # Prepare sequences
#     X, y = [], []
#     DAYS_LOOKBACK = 1
#     for i in range(DAYS_LOOKBACK, len(scaled_data) - DAYS_LOOKBACK):
#         X.append(scaled_data[i-DAYS_LOOKBACK:i])
#         y.append(scaled_data_Y[i])
    
#     # Log data statistics for both outputs
#     logger.info(f"Target statistics:")
#     logger.info(f"Return - Mean: {np.mean(scaled_data_Y[:, 0]):.4f}, Std: {np.std(scaled_data_Y[:, 0]):.4f}")
#     logger.info(f"StdDev - Mean: {np.mean(scaled_data_Y[:, 1]):.4f}, Std: {np.std(scaled_data_Y[:, 1]):.4f}")
    
#     return np.array(X), np.array(y)  # y will now be (n_samples, 2)




DATA_FOLDER = "../data/"
TRAIN_SIZE_SPLIT = 0.9

# Wrap the main execution code
if __name__ == '__main__':
    data_model = DataModel()
    data_model.get_data()