import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import boto3
import io
import os
import logging
from torch.utils.data import DataLoader, TensorDataset

# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_torch():
    # Check if MPS (Mac GPU) or CUDA is available and set the device
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f'Using device: {device}')

    # Create a simple XOR dataset and move to device
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32).to(device)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

    # Define a simple feedforward neural network 
    class Net(nn.Module): 
        def __init__(self): 
            super(Net, self).__init__() 
            self.fc1 = nn.Linear(2, 8) 
            self.fc2 = nn.Linear(8, 8) 
            self.fc2 = nn.Linear(8, 1) 
            
        def forward(self, x): 
            x = torch.relu(self.fc1(x)) 
            x = torch.relu(x)
            x = torch.sigmoid(self.fc2(x)) 
            return x 
        
    # Initialize the network and move to device
    net = Net().to(device)

    # Define a loss function and optimizer 
    criterion = nn.MSELoss() 
    optimizer = optim.SGD(net.parameters(), lr=0.01) 

    # Train the network 
    for epoch in range(50000): 
        optimizer.zero_grad() 
        output = net(X) 
        loss = criterion(output, y) 
        loss.backward() 
        optimizer.step()
        
        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Test the network 
    with torch.no_grad(): 
        output = net(X) 
        print(output) 

def load_data(TICKERS, DATA_FOLDER):
    logger.info(f"Starting data loading for {len(TICKERS)} tickers from {DATA_FOLDER}")
    all_data = {}
    for ticker in TICKERS: 
        logger.debug(f"Loading data for {ticker}")
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
    logger.info("Data loading completed successfully")
    return all_data
# Preprocessing function
def preprocess_data(df):
    logger.debug("Starting data preprocessing")
    # Add return and month columns
    df = df.sort_index( ascending = True )
    df['Return'] = df['Close'].pct_change()
    df['Price_Diff'] = df['Close'].diff()
    df['Next_Day_Return'] = df['Return'].shift(-1)

    # Remove outliers 
    df = df[(np.abs(df['Return']) < 0.1)]

    # Add total return for the next 7 days
    df['Next_7_Day_Return'] = df['Return'].rolling(window=7).sum().shift(-6)
    df['Next_7_Day_Return_StdDev'] = df['Return'].rolling(window=7).std().shift(-6)
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
        'Return','Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', 
        'Lagged_Return_4', 'Lagged_Return_5', 'Lagged_Return_6', 'Lagged_Return_7',
        'RSI', 'OBV',
        'Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 
        'Is_June', 'Is_July', 'Is_August', 'Is_September', 
        'Is_October', 'Is_November', 'Is_December',
        'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday',
        'Is_2020', 'Is_2019', 'Is_2018', 'Is_2017', 'Is_2016', 'Is_2015', 'Is_2014', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024',
        # 'Random'
        ]]
    )
    scaled_data_Y = np.array(df[['Next_Day_Return', 'Next_7_Day_Return_StdDev']])

    # Prepare sequences
    X, y = [], []
    DAYS_LOOKBACK = 1
    for i in range(DAYS_LOOKBACK, len(scaled_data) - DAYS_LOOKBACK):
        X.append(scaled_data[i-DAYS_LOOKBACK:i])
        y.append(scaled_data_Y[i])
    
    # Log data statistics for both outputs
    logger.info(f"Target statistics:")
    logger.info(f"Return - Mean: {np.mean(scaled_data_Y[:, 0]):.4f}, Std: {np.std(scaled_data_Y[:, 0]):.4f}")
    logger.info(f"StdDev - Mean: {np.mean(scaled_data_Y[:, 1]):.4f}, Std: {np.std(scaled_data_Y[:, 1]):.4f}")
    
    return np.array(X), np.array(y)  # y will now be (n_samples, 2)


# Define the neural network for stock prediction
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(38, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 2)  # Output single value for return
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)  # No activation for regression output
        return x

TICKERS = ['GOOG', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA',
'UNH', 'XOM', 'LLY', 'JPM', 'V', 'PG', 'AVGO', 'MRK', 'COST', 'PEP', 'ADBE', 
'HUM', 'ELV', 'CVS', 'CNC', 'CI',
'ALL', 'AXP', 'AIG', 'AMGN', 'AON', 'T', 'BKR', 'BBY', 'BA', 'COF', 'CSCO'
]
DATA_FOLDER = "data/"
TRAIN_SIZE_SPLIT = 0.9

# Wrap the main execution code
if __name__ == '__main__':
    logger.info("Starting stock prediction model training")
    
    # Load and preprocess data
    all_data = load_data(TICKERS, DATA_FOLDER)
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

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_all),
        torch.FloatTensor(y_train_all)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_all),
        torch.FloatTensor(y_test_all)
    )

    # Create data loaders with num_workers=0 initially
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=5  # Set to 0 for debugging
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=5  # Set to 0 for debugging
    )

    # Set up device
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f'Using device: {device}')

    # Create and train model
    model = StockPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(50):
        model.train()
        total_loss_return = 0
        total_loss_stddev = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            
            # Split the loss calculation for return and stddev
            loss_return = criterion(output[:, 0].unsqueeze(1), batch_y[:, 0].unsqueeze(1))
            loss_stddev = criterion(output[:, 1].unsqueeze(1), batch_y[:, 1].unsqueeze(1))
            loss = loss_return + loss_stddev  # You can adjust the weighting if needed
            
            loss.backward()
            optimizer.step()
            
            total_loss_return += loss_return.item()
            total_loss_stddev += loss_stddev.item()
            batch_count += 1
        
        avg_loss_return = total_loss_return / batch_count
        avg_loss_stddev = total_loss_stddev / batch_count
        logger.info(f'Epoch {epoch}, Return Loss: {avg_loss_return:.6f}, StdDev Loss: {avg_loss_stddev:.6f}')

    # Testing phase
    logger.info("Starting testing phase...")
    model.eval()
    total_abs_error_return = 0
    total_abs_error_stddev = 0
    total_squared_error_return = 0
    total_squared_error_stddev = 0
    total_samples = 0
    all_predictions_return = []
    all_predictions_stddev = []
    all_actuals_return = []
    all_actuals_stddev = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            
            # Calculate errors separately for return and stddev
            abs_error_return = torch.abs(predictions[:, 0] - batch_y[:, 0])
            abs_error_stddev = torch.abs(predictions[:, 1] - batch_y[:, 1])
            squared_error_return = (predictions[:, 0] - batch_y[:, 0]) ** 2
            squared_error_stddev = (predictions[:, 1] - batch_y[:, 1]) ** 2
            
            total_abs_error_return += abs_error_return.sum().item()
            total_abs_error_stddev += abs_error_stddev.sum().item()
            total_squared_error_return += squared_error_return.sum().item()
            total_squared_error_stddev += squared_error_stddev.sum().item()
            total_samples += batch_y.size(0)
            
            # Store predictions and actuals
            all_predictions_return.extend(predictions[:, 0].cpu().numpy())
            all_predictions_stddev.extend(predictions[:, 1].cpu().numpy())
            all_actuals_return.extend(batch_y[:, 0].cpu().numpy())
            all_actuals_stddev.extend(batch_y[:, 1].cpu().numpy())
    
    # Calculate metrics for both outputs
    mae_return = total_abs_error_return / total_samples
    mae_stddev = total_abs_error_stddev / total_samples
    mse_return = total_squared_error_return / total_samples
    mse_stddev = total_squared_error_stddev / total_samples
    rmse_return = np.sqrt(mse_return)
    rmse_stddev = np.sqrt(mse_stddev)
    
    # Log results
    logger.info("\nTest Results:")
    logger.info("Return Prediction Metrics:")
    logger.info(f"  MAE: {mae_return:.6f}")
    logger.info(f"  RMSE: {rmse_return:.6f}")
    logger.info("StdDev Prediction Metrics:")
    logger.info(f"  MAE: {mae_stddev:.6f}")
    logger.info(f"  RMSE: {rmse_stddev:.6f}")
    
    # Save model and metrics
    save_dict = {
        'model_state': model.state_dict(),
        'test_metrics': {
            'mae_return': mae_return,
            'mse_return': mse_return,
            'rmse_return': rmse_return,
            'mae_stddev': mae_stddev,
            'mse_stddev': mse_stddev,
            'rmse_stddev': rmse_stddev
        }
    }

    try:
        # Save model and test metrics
        buffer = io.BytesIO()
        torch.save(save_dict, buffer)
        buffer.seek(0)
        
        s3_client = boto3.client('s3')
        bucket_name = 'tradingmodelsahmed'
        s3_key = 'models/model_with_metrics.pth'
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        logger.info(f"\nModel and metrics successfully saved to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Error saving model to S3: {str(e)}")

logger.info("Process completed successfully")
