from openai import OpenAI, RateLimitError
import json
import subprocess
import tempfile

openai_client = OpenAI(api_key='sk-proj-rMn1qOJB9Yx2hPNJOryJT3BlbkFJAUa9pyzuZvbkpPnnfin3')

# Function to get improvement suggestions for the code
def get_code_suggestions(script: str, output: str, model: str = "gpt-4") -> str:
    prompt = f"""
    Here's a Python script for building an LSTM model to predict financial data performance:
    
    {script}
    
    The script produced the following output when executed:
    
    {output}
    
    Suggest specific code edits to improve the model's performance. The goal is to bring down MSE and other indicators of model robustnuess. 
    Focus on:
    - Optimizing the LSTM architecture
    - Hyperparameter tuning
    - Improving data preprocessing and handling
    
    Provide the suggestions in clear and very specific steps.
    """
    response_chunk = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=0,
    )        
    response = response_chunk.choices[0].message.content
    return response

# Function to generate updated code from the suggestions
def get_updated_code(script: str, suggestions: str, model: str = "gpt-4") -> str:
    prompt = f"""
    Here's a Python script for building an LSTM model to predict financial data performance:
    
    {script}
    
    Based on the following suggestions, update the script to improve it:
    
    {suggestions}
    
    Return the code as JSON in the following format:
    {{"code": "..."}}.
    """
    response_chunk = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=0,
    )        
    response = response_chunk.choices[0].message.content
    return json.loads(response)["code"]

# Function to execute the script and capture its output
def execute_script(script: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_script:
        temp_script.write(script.encode('utf-8'))
        temp_script_path = temp_script.name
    
    try:
        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True,
            text=True
        )
        output = result.stdout + result.stderr
    finally:
        # Clean up the temporary file
        import os
        os.remove(temp_script_path)
    
    return output

# Main workflow loop
def iterative_refinement(initial_script: str, iterations: int = 5):
    script = initial_script
    for i in range(iterations):
        print(f"--- Iteration {i+1} ---")
        
        # Execute the script and capture output
        output = execute_script(script)
        print("Script Output:\n", output)
        
        # Get improvement suggestions
        suggestions = get_code_suggestions(script, output)
        print("Suggestions:\n", suggestions)
        
        # Generate updated code
        updated_script = get_updated_code(script, suggestions)
        print("Updated Script:\n", updated_script)
        
        # Update the script for the next iteration
        script = updated_script

    return script

# Example usage
initial_script = """from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



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
    # Print 5 rows of data to check
    all_data[ticker] = data

# Preprocessing function
def preprocess_data(df):
    # Add return and month columns
    df['Return'] = df['Close'].pct_change()
    df['Month'] = df.index.month
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

    # Drop NaN values
    df.dropna(inplace=True)

    # Scale data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(df[['Return', 'Month']])
    scaled_data = np.array(df[['Close', 'Low', 'Open', 'High', 'annual_volatility', 'Month',  '7_day_avg','30_day_avg', '30_day_min', '30_day_max']])
    
    # Prepare sequences
    X, y = [], []
    DAYS_LOOKBACK = 30
    DAYS_FORWARD = 1
    for i in range(DAYS_LOOKBACK, len(scaled_data) - DAYS_LOOKBACK):
        # Input: last 7 days (price, return, month)
        X.append(scaled_data[i-DAYS_LOOKBACK:i])
        
        # Output: return for the next 7 days
        y.append(np.average(scaled_data[i:i+DAYS_FORWARD, 0]))  # Return column index is 1
    
    return np.array(X), np.array(y) #, scaler

def build_model (X_train, y_train, X_test, y_test): 
    # Build the LSTM model
    new_model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(100),
        Dense(1)  # Output size is 7 (returns for the next 7 days)
    ])
    new_model.compile(optimizer='adam', loss='mean_squared_error')
    print("Training the model...")
    # Train the model
    new_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    # Inverse transform predictions and actual values (for the 'Return' column only)
    # predictions_rescaled = scaler.inverse_transform(
    #     np.hstack([predictions.reshape(-1, 1), np.zeros((predictions.shape[0], 1))])
    # )[:, 0]  # Extract only the 'Return' column
    # y_test_rescaled = scaler.inverse_transform(
    #     np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))])
    # )[:, 0]  # Extract only the 'Return' column
    return new_model

# Prepare data for one ticker (e.g., AAPL)
ticker = "AAPL"
train_or_load = "train"

df = all_data[ticker]
# X, y, scaler = preprocess_data(df)
X, y = preprocess_data(df)

# Split data into training and test sets
train_size = int(0.98 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

if train_or_load == "train":
    model = build_model (X_train, y_train, X_test, y_test)
    model.save(f'models/{ticker}_LSTM_v1.keras')

else:
    model = tf.keras.models.load_model(f'models/{ticker}_LSTM_v1.keras')


# Get MSE of the model
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error for testing: {mse}")

# Make predictions
predictions = model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error for training: {mse}")
"""
final_script = iterative_refinement(initial_script)
print("Final Script:", final_script)
