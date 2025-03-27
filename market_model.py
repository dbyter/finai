from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight




def load_data (DATA_FOLDER, TICKERS):
    DATA_FOLDER = "data/"
    # List of stock symbols to download
    all_data = {}
    for ticker in TICKERS: 
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
        # Print 5 rows of data to check
        all_data[ticker] = data
    return all_data

# Preprocessing function
def preprocess_data(df):
    # Add return and month columns
    df = df.sort_index( ascending = True )
    df['Return'] = df['Close'].pct_change()
    df['Price_Diff'] = df['Close'].diff()
    df['Next_Day_Return'] = df['Return'].shift(-1)

    # Remove outliers 
    df = df[(np.abs(df['Return']) < 0.1)]

    # Add total return for the next 7 days
    df['Next_7_Day_Return'] = df['Return'].rolling(window=7).sum().shift(-6)
    # Create classification labels for 7 day return 
    # Highly positive: > 2%
    # Positive: 0% to 2%
    # Negative: 0% to -2%
    # Highly negative: < -2%
    df['Highly_Positive_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if x > 0.02 else 0)
    df['Highly_Negative_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if x < -0.02 else 0)
    df['Positive_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if 0.02 >= x > 0 else 0)
    df['Negative_7'] = df['Next_7_Day_Return'].apply(lambda x: 1 if 0 > x >= -0.02 else 0)

    # Get correlation of next day return with other features
    # Lagged return
    df['Lagged_Return_1'] = df['Return'].shift(1)
    df['Lagged_Return_2'] = df['Return'].shift(2)
    df['Lagged_Return_3'] = df['Return'].shift(3)
    df['Lagged_Return_4'] = df['Return'].shift(4)
    df['Lagged_Return_5'] = df['Return'].shift(5)
    df['Lagged_Return_6'] = df['Return'].shift(6)
    df['Lagged_Return_7'] = df['Return'].shift(7)
    
    # Plot the acf chart 


    df['Month'] = df.index.month
    # Create a binary column to indicate the month
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

    # Day of the week
    df['Day_of_week'] = df.index.dayofweek
    # Create a binary column to indicate the day of the week
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

    # Scale data
    scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(df[['Return', 'Month']])
    # scaled_data = scaler.fit_transform(df[['Return', 'Month', 'Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', '30_day_avg_return', '7_day_avg_return', 
    # '30_day_min_return', '30_day_max_return', 'RSI', 'Volume', 'OBV']])
    scaled_data = scaler.fit_transform(
        df[[
        'Return', 'RSI', 'OBV',
        'Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 
        'Is_June', 'Is_July', 'Is_August', 'Is_September', 
        'Is_October', 'Is_November', 'Is_December',
        'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday',
        'Is_2020', 'Is_2019', 'Is_2018', 'Is_2017', 'Is_2016', 'Is_2015', 'Is_2014', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024',
        # 'Random'
        ]]
    )
    scaled_data_Y = np.array(df[['Highly_Positive', 'Positive', 'Negative', 'Highly_Negative']])

    # Prepare sequences
    X, y = [], []
    DAYS_LOOKBACK = 1
    DAYS_FORWARD = 1
    for i in range(DAYS_LOOKBACK, len(scaled_data) - DAYS_LOOKBACK):
        # Input: last 7 days (price, return, month)
        X.append(scaled_data[i-DAYS_LOOKBACK:i])
        
        # Output: return for the next 7 days
        y.append(scaled_data_Y[i])  

    
    return np.array(X), np.array(y) #, scaler

def build_model (X_train, y_train, X_test, y_test): 
    """Builds and trains an LSTM model using the given training and test data."""
    
    def build (X_train, y_train, X_test, y_test):
        new_model = Sequential([
            LSTM(100,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            ),
            LSTM(100,
            ),
            Dense(1)
        ])
        new_model.compile(optimizer='adam', loss='mean_squared_error')
        print("Training the model...")
        new_model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test))
        return new_model
    def test (model, X_test, y_test):
        predictions = model.predict(X_test)
        # Calculate the mean squared error of the predictions
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Mean Squared Error for training: {mse}")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, train_size+1), y_train, label="Train data", color="blue")
        plt.plot(np.arange(train_size, train_size + predictions.shape[0]), predictions, label="Predictions", color="red")
        plt.plot(np.arange(train_size, train_size + predictions.shape[0]), y_test, label="Actual", color="orange", linestyle="--")
        plt.title("Predictions vs Actual Returns")
        plt.xlabel("Sample Index")
        plt.ylabel("Return")
        plt.legend()
        plt.show()
    
    model = build(X_train, y_train, X_test, y_test)
    test(model, X_test, y_test)


def build_model_categorize (X_train, y_train, X_test, y_test):
    """Builds and trains an LSTM model using the given training and test data."""
    def build (X_train, y_train, X_test, y_test): 
        # Build the LSTM model
        new_model = Sequential([
            Input (shape = (X_train.shape[1], X_train.shape[2])),
            LSTM (8),
            Dense (8, activation='relu'),
            Dense (8, activation='relu'),
            Dense (8, activation='relu'),
            Dense (8, activation='relu'),
            Dense (8, activation='relu'),
            Dense (4, activation='relu'),
            Dense (4, activation='softmax')
        ])
        all_classes = np.arange(y_train.shape[1])  # If y_train is one-hot encoded
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=all_classes, 
            y=y_train.argmax(axis=1)  # Convert one-hot to class labels
        )
        class_weight_dict = dict(enumerate(class_weights))
        new_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Training the model...")
        # Train the model
        new_model.fit(X_train,  y_train, class_weight = class_weight_dict,  epochs=150, batch_size=32, validation_data=(X_test, y_test))
        print(new_model.summary())
        return new_model

    def test_model (model, X_test, y_test):
        predictions = model.predict(X_test)
        # Convert one-hot encoded predictions and y_test to category indices
        pred_categories = np.argmax(predictions, axis=1)
        true_categories = np.argmax(y_test, axis=1)

        # Count the occurrences of each category
        unique_categories = np.unique(np.concatenate([pred_categories, true_categories]))
        pred_counts = [np.sum(pred_categories == cat) for cat in unique_categories]
        true_counts = [np.sum(true_categories == cat) for cat in unique_categories]

        # Create bar chart
        plt.figure(figsize=(10, 6))
        x = np.arange(len(unique_categories))
        width = 0.35

        plt.bar(x - width/2, true_counts, width, label='Actual', color='blue', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predictions', color='red', alpha=0.7)

        plt.xlabel('Category')
        plt.ylabel('Frequency')
        plt.title('Comparison of Actual vs Predicted Categories')
        plt.xticks(x, unique_categories)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    model = build(X_train, y_train, X_test, y_test)
    test_model(model, X_test, y_test)
    return model

def build_model_categorize_GB (X_train, y_train, X_test, y_test):
    """Train a model using Gradient Boosting for categorization."""
    def build (X_train, y_train, X_test, y_test): 
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.utils import class_weight

        model = GradientBoostingClassifier(n_estimators=150, max_depth=3, verbose = 1)
        y_train_nothot = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train_nothot)

        model.fit(np.squeeze(X_train, axis = 1), y_train_nothot, sample_weight=class_weights)
        return model
    
    def test_model (model, X_test, y_test):
        predictions = model.predict(np.squeeze(X_test, axis =1))
        # Convert one-hot encoded predictions and y_test to category indices
        pred_categories = predictions
        true_categories = np.argmax(y_test, axis =1 )
        print(pred_categories[:100])
        print(true_categories[:100])

        # Count the occurrences of each category
        unique_categories = np.unique(np.concatenate([pred_categories, true_categories]))
        pred_counts = [np.sum(pred_categories == cat) for cat in unique_categories]
        true_counts = [np.sum(true_categories == cat) for cat in unique_categories]

        # Calculate the % of correct predictions
        correct = np.sum(pred_categories == true_categories)
        total = len(true_categories)
        print(f"Correct predictions: {correct} / {total} ({correct/total:.2%})")
        # Create bar chart
        plt.figure(figsize=(10, 6))
        x = np.arange(len(unique_categories))
        width = 0.35

        plt.bar(x - width/2, true_counts, width, label='Actual', color='blue', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predictions', color='red', alpha=0.7)

        plt.xlabel('Category')
        plt.ylabel('Frequency')
        plt.title('Comparison of Actual vs Predicted Categories')
        plt.xticks(x, unique_categories)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    model = build(X_train, y_train, X_test, y_test)
    test_model(model, X_test, y_test)


def model_builder (X_train, y_train, X_test, y_test, model_type = "regression"):
    if model_type == "regression":
        return build_model(X_train, y_train, X_test, y_test)
    else:
        return build_model_categorize_GB(X_train, y_train, X_test, y_test)

def single_ticker_run (TICKER, TRAIN_OR_LOAD, MODEL_TO_USE, TRAIN_SIZE_SPLIT):
    df = all_data[TICKER]
    X, y = preprocess_data(df)
    train_size = int(TRAIN_SIZE_SPLIT * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if TRAIN_OR_LOAD == "train":
        model = model_builder (X_train, y_train, X_test, y_test, MODEL_TO_USE)
        model.save(f'models/{TICKER}_LSTM_v1.keras')
    else:
        model = tf.keras.models.load_model(f'models/{TICKER}_LSTM_v1.keras')

    return model

def multi_ticker_run (TICKERS, TRAIN_OR_LOAD, MODEL_TO_USE, TRAIN_SIZE_SPLIT, all_data):
    X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []

    for ticker in TICKERS:
        df = all_data[ticker]
        X, y = preprocess_data(df)
        train_size = int(TRAIN_SIZE_SPLIT * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        y_train_all.append(y_train)
        y_test_all.append(y_test)
    
    X_train_all = np.concatenate(X_train_all)
    X_test_all = np.concatenate(X_test_all)
    y_train_all = np.concatenate(y_train_all)
    y_test_all = np.concatenate(y_test_all)

    if TRAIN_OR_LOAD == "train":
        print("Training model... with # of observations: ", len(X_train_all))
        model = model_builder (X_train_all, y_train_all, X_test_all, y_test_all, MODEL_TO_USE)
        model.save(f'models/multi_ticker_LSTM_v1.keras')
    else:
        model = tf.keras.models.load_model(f'models/multi_ticker_LSTM_v1.keras')


    return models

if __name__ == "__main__":
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    TICKERS = ['GOOG', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA',
    'UNH', 'XOM', 'LLY', 'JPM', 'V', 'PG', 'AVGO', 'MRK', 'COST', 'PEP', 'ADBE', 
    'HUM', 'ELV', 'CVS', 'CNC', 'CI',
    'ALL', 'AXP', 'AIG', 'AMGN', 'AON', 'T', 'BKR', 'BBY', 'BA', 'COF', 'CSCO'
    ]
    DATA_FOLDER = "data/"
    all_data = load_data(DATA_FOLDER, TICKERS)

    # CONFIGURATION
    TICKER = "AAPL"
    TRAIN_OR_LOAD = "train"
    MODEL_TO_USE = "categorize"
    TRAIN_SIZE_SPLIT = 0.9
    # single_ticker_run(TICKER, TRAIN_OR_LOAD, MODEL_TO_USE, TRAIN_SIZE_SPLIT)

    TRAIN_OR_LOAD = "train"
    MODEL_TO_USE = "categorize"
    TRAIN_SIZE_SPLIT = 0.9
    multi_ticker_run(TICKERS, TRAIN_OR_LOAD, MODEL_TO_USE, TRAIN_SIZE_SPLIT, all_data)



