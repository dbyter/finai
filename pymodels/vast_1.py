import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import boto3
import io
import logging
from torch.utils.data import DataLoader, TensorDataset
from data_model import DataModel
import torch.nn.functional as F
from torch.distributions import Normal

# Set up logging configuration at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def create_sequences(df, train_start_date, train_end_date, test_start_date, test_end_date, hotencode_features, hotencode_key):
    logger.debug("Starting data preprocessing")
    scaler = StandardScaler()
    # df [hotencode_features] = 0
    # df [f'hotencode_ticker_{hotencode_key}'] = 1
    df ['assets_liabilities_ratio'] = df['total_assets'] / df['total_liabilities']
    df.dropna()
    X_df = df[[
        'Return','Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', 
        'Lagged_Return_4', 'Lagged_Return_5', 'Lagged_Return_6', 'Lagged_Return_7',
        'RSI', 'OBV',
        'Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 
        'Is_June', 'Is_July', 'Is_August', 'Is_September', 
        'Is_October', 'Is_November', 'Is_December',
        'Is_Monday', 'Is_Tuesday', 'Is_Wednesday', 'Is_Thursday', 'Is_Friday',
        'Is_2020', 'Is_2019', 'Is_2018', 'Is_2017', 'Is_2016', 'Is_2015', 'Is_2014', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024',
        'Random', 'Date', '7_day_max', '7_day_min', '30_day_max', '30_day_min', '30_day_avg', '30_day_avg_return', '7_day_avg', '7_day_avg_return'
        ]].copy()
    
    train_X_df = X_df[X_df['Date'] < train_end_date].drop(columns=['Date'])
    test_X_df = X_df[X_df['Date'] >= train_end_date].drop(columns=['Date'])
    train_X_df_scaled = scaler.fit_transform(train_X_df)
    test_X_df_scaled = scaler.fit_transform(test_X_df)

    Y_df = df[['Next_Day_Return', 'Next_7_Day_Return_StdDev', 'Date']]
    train_Y_df = Y_df[Y_df['Date'] < train_end_date].drop(columns=['Date'])
    test_Y_df = Y_df[Y_df['Date'] >= train_end_date].drop(columns=['Date'])
    train_Y_df_scaled = scaler.fit_transform(train_Y_df)
    test_Y_df_scaled = scaler.fit_transform(test_Y_df)

    return train_X_df_scaled, test_X_df_scaled, train_Y_df_scaled, test_Y_df_scaled

    # print(type(train_X_df_scaled))
    # scaled_data_Y = np.array(df[['Next_Day_Return', 'Next_7_Day_Return_StdDev']])

    # # Prepare sequences
    # X, y = [], []
    # DAYS_LOOKBACK = 1
    # for i in range(DAYS_LOOKBACK, len(scaled_data) - DAYS_LOOKBACK):
    #     X.append(scaled_data[i-DAYS_LOOKBACK:i])
    #     y.append(scaled_data_Y[i])


# Define the neural network for stock prediction
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(47, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  # No activation for regression output
        return x



DATA_FOLDER = "../data/"

train_start_date = "2015-01-01"
train_end_date = "2024-01-01"
test_start_date = "2025-01-01"
test_end_date = "2025-12-31"

# Wrap the main execution code
if __name__ == '__main__':
    logger.info("Starting stock prediction model training")
    
    # Load and preprocess data
    d = DataModel()
    all_data = d.get_data()
    all_data = {k:v for k,v in all_data.items() if k in ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'GOOG', 'META', 'NFLX', 'TSM', 'WMT']}
    X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []
    # all_data = all_data.reset_index(inplace=True)
    # print(list(all_data.keys()))
    hotencode_key = 0
    # Create a feature for each ticker
    N_tickers = len(all_data.keys())
    hotencode_features = [f'hotencode_ticker_{i}' for i in range(N_tickers)]
    for ticker, df in all_data.items():
        try:
            # Create a hot encoded ticker column
            X_train, X_test, y_train, y_test = create_sequences(df, train_start_date, train_end_date, test_start_date, test_end_date, hotencode_features, hotencode_key)
            X_train_all.append(X_train)
            X_test_all.append(X_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)
        except Exception as e:
            logger.error(f"Error processing {ticker}: {len(df)} {e}")
            continue
        hotencode_key += 1

    X_train_all = np.concatenate(X_train_all)
    X_test_all = np.concatenate(X_test_all)
    y_train_all = np.concatenate(y_train_all)
    y_test_all = np.concatenate(y_test_all)

    print(X_train_all.shape)
    print(y_train_all.shape)
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
    device = torch.device("cpu")
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
            print(type(loss))
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

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 2)  # Output mean and std for action distribution
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        mean = dist[:, 0]
        std = F.softplus(dist[:, 1]) + 1e-5  # Ensure std is positive
        return mean, std

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

class PPOAgent:
    def __init__(self, input_dims, alpha=0.0003, gamma=0.99, n_epochs=10, batch_size=64,
                 epsilon=0.2, fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size)
        
        self.actor = ActorNetwork(input_dims, alpha, fc1_dims, fc2_dims)
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims, fc2_dims)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.FloatTensor(observation).to(self.actor.device)
        action, log_prob = self.actor.get_action(state)
        value = self.critic(state)
        
        return action, log_prob, value.detach().cpu().numpy()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-dones_arr[k]) - values[k])
                    discount *= self.gamma*0.95
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch]).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                mean, std = self.actor(states)
                dist = Normal(mean, std)
                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp(new_probs - old_probs)

                weighted_probs = prob_ratio * advantage[batch]
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_value = self.critic(states)
                critic_loss = F.mse_loss(critic_value, values[batch])

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
