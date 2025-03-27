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

    logger.debug("Data preprocessing completed")
    return np.array(X), np.array(y) #, scaler


# Define the neural network for stock prediction
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input sequence
        self.fc1 = nn.Linear(38, 100)  # Input features -> first hidden layer
        self.fc2 = nn.Linear(100, 100)  # First -> second hidden layer
        self.fc3 = nn.Linear(100, 30)  # First -> second hidden layer
        self.fc4 = nn.Linear(30, 4)   # Second hidden layer -> output (4 classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x= self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x

TICKERS = ['GOOG', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA',
'UNH', 'XOM', 'LLY', 'JPM', 'V', 'PG', 'AVGO', 'MRK', 'COST', 'PEP', 'ADBE', 
'HUM', 'ELV', 'CVS', 'CNC', 'CI',
'ALL', 'AXP', 'AIG', 'AMGN', 'AON', 'T', 'BKR', 'BBY', 'BA', 'COF', 'CSCO'
]
DATA_FOLDER = "data/"
TRAIN_SIZE_SPLIT = 0.9

logger.info("Starting stock prediction model training")

logger.info("Loading data...")
all_data = load_data(TICKERS, DATA_FOLDER)
X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []

logger.info("Preprocessing data for each ticker...")
for ticker in TICKERS:
    logger.debug(f"Processing {ticker}")
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

logger.info(f"Data shapes - Training: {X_train_all.shape}, Testing: {X_test_all.shape}")

logger.info("Converting data to PyTorch tensors...")
X_train_tensor = torch.FloatTensor(X_train_all)
y_train_tensor = torch.FloatTensor(y_train_all)
X_test_tensor = torch.FloatTensor(X_test_all)
y_test_tensor = torch.FloatTensor(y_test_all)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
logger.info(f'Using device: {device}')

logger.info("Moving data to device...")
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
model = StockPredictor().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger.info("Starting training...")
for epoch in range(50000): 
    optimizer.zero_grad() 
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward() 
    optimizer.step()
    
    if epoch % 1000 == 0:
        logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')

logger.info("Training completed. Saving model...")

try:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    s3_client = boto3.client('s3')
    bucket_name = os.environ['S3_BUCKET_NAME']
    s3_key = 'models/model.pth'
    s3_client.upload_fileobj(buffer, bucket_name, s3_key)
    logger.info(f"Model successfully saved to s3://{bucket_name}/{s3_key}")
except Exception as e:
    logger.error(f"Error saving model to S3: {str(e)}")

logger.info("Process completed successfully")
