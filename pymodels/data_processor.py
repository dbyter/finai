import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, Dict, List
import logging
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data import MultiNormalizer

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the dataframe with basic preprocessing"""
        logger.info(f"Original data shape: {df.shape}")
        df = df.copy()
        
        # Log date range
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Check for missing values before processing
        missing_before = df.isnull().sum()
        
        # Calculate ratio only where both values are present
        mask = df['total_assets'].notna() & df['total_liabilities'].notna()
        df.loc[mask, 'assets_liabilities_ratio'] = df.loc[mask, 'total_assets'] / df.loc[mask, 'total_liabilities']
        
        # Calculate Return (daily return)
        df['Return'] = df['Close'].pct_change()
        
        # Calculate Volume (normalized)
        df['Volume'] = df['Volume'].fillna(0)
        df['Volume'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        
        # Add year indicators
        for year in range(2020, 2026):
            df[f'Is_{year}'] = (df['Date'].dt.year == year).astype(int)
        
        # Calculate dependent variables
        df['Next_Day_Return'] = df['Return'].shift(-1)
        df['Next_7_Day_Return'] = df['Return'].rolling(window=7, min_periods=1).mean().shift(-7)
        df['Next_7_Day_Return_StdDev'] = df['Return'].rolling(window=7, min_periods=1).std().shift(-7)
        
        # Log missing values after feature calculation
        missing_after_features = df.isnull().sum()
        
        # Drop rows only for specific columns that are critical for the model
        critical_columns = self.config.FEATURES + self.config.DEPENDENT_VARIABLES
        df = df.dropna(subset=critical_columns)
        
        logger.info(f"Data shape after dropping missing values in critical columns: {df.shape}")
        
        if df.shape[0] == 0:
            raise ValueError("No data remains after preprocessing. Please check the data quality and required columns.")
            
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for the basic model"""
        logger.debug("Starting data preprocessing")
        
        # Add hot encoding
        # df[hotencode_features] = 0
        # df[f'hotencode_ticker_{hotencode_key}'] = 1
        
        # Split data
        df_train = df[df['Date'] < self.config.TRAIN_END_DATE]
        df_test = df[df['Date'] >= self.config.TRAIN_END_DATE]
        
        logger.info(f"Train data shape: {df_train.shape}")
        logger.info(f"Test data shape: {df_test.shape}")
        
        # Prepare features and targets
        train_X_df = df_train[self.config.FEATURES].copy()
        test_X_df = df_test[self.config.FEATURES].copy()
        train_Y_df = df_train[self.config.DEPENDENT_VARIABLES].copy()
        test_Y_df = df_test[self.config.DEPENDENT_VARIABLES].copy()
        
        logger.info(f"Train X shape: {train_X_df.shape}")
        logger.info(f"Train Y shape: {train_Y_df.shape}")
        logger.info(f"Test X shape: {test_X_df.shape}")
        logger.info(f"Test Y shape: {test_Y_df.shape}")
        
        # Check for any remaining missing values
        logger.info(f"Missing values in train X: {train_X_df.isnull().sum().sum()}")
        logger.info(f"Missing values in train Y: {train_Y_df.isnull().sum().sum()}")
        
        # Scale data
        train_X_scaled = self.scaler.fit_transform(train_X_df)
        test_X_scaled = self.scaler.transform(test_X_df)
        train_Y_scaled = self.scaler.fit_transform(train_Y_df)
        test_Y_scaled = self.scaler.transform(test_Y_df)
        
        return train_X_scaled, test_X_scaled, train_Y_scaled, test_Y_scaled

    def create_sequences_tft(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def lagged_sequences(features, targets, lookback, horizon=1):
            """
            X : window [t-lookback+1, …, t]         (length = lookback)
            y : target at t + horizon               (one step ahead if horizon=1)
            """
            Xs, ys = [], []
            end = len(features) - horizon           # last usable t
            for t in range(lookback, end):
                Xs.append(features[t - lookback + 1 : t + 1]) # ends at t-1 when horizon=1
                ys.append(targets[t + horizon])     # label at t+horizon
            return np.array(Xs), np.array(ys)
            
        logger.debug("Starting data preprocessing")
        
        # Split data
        train_mask = df['Date'] < self.config.TRAIN_END_DATE
        test_mask = df['Date'] >= self.config.TRAIN_END_DATE
        
        feature_scaler = StandardScaler()
        target_scaler  = StandardScaler()

        # Scale features and targets separately
        train_x_scaled = feature_scaler.fit_transform(df[train_mask][self.config.FEATURES])
        test_x_scaled = feature_scaler.transform(df[test_mask][self.config.FEATURES])
        train_y_scaled = target_scaler.fit_transform(df[train_mask][self.config.DEPENDENT_VARIABLES])
        test_y_scaled = target_scaler.transform(df[test_mask][self.config.DEPENDENT_VARIABLES])
        
        # Create multi-step sequences
        train_X, train_Y = lagged_sequences(train_x_scaled, train_y_scaled, 7, 1)
        test_X, test_Y = lagged_sequences(test_x_scaled, test_y_scaled, 7, 1)

        # logger.info(f"Train X shape: {train_X.shape}")
        # logger.info(f"Train Y shape: {train_Y.shape}")
        # logger.info(f"Test X shape: {test_X.shape}")
        # logger.info(f"Test Y shape: {test_Y.shape}")
        
        return train_X, test_X, train_Y, test_Y

    def create_sequences_tft_with_ticker_scaling(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences with per-ticker scaling and hot encoding.
        Each ticker gets its own scaler for features and targets, and hot encoding is added without scaling.
        """
        def lagged_sequences(features, targets, lookback, horizon=1):
            """
            X : window [t-lookback+1, …, t]         (length = lookback)
            y : target at t + horizon               (one step ahead if horizon=1)
            """
            Xs, ys = [], []
            end = len(features) - horizon           # last usable t
            for t in range(lookback, end):
                Xs.append(features[t - lookback + 1 : t + 1]) # ends at t-1 when horizon=1
                ys.append(targets[t + horizon])     # label at t+horizon
            return np.array(Xs), np.array(ys)
            
        logger.debug("Starting data preprocessing with ticker-specific scaling")
        
        # Create hot encoding for tickers
        ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
        df = pd.concat([df, ticker_dummies], axis=1)
        
        # Split data
        train_mask = df['Date'] < self.config.TRAIN_END_DATE
        test_mask = df['Date'] >= self.config.TRAIN_END_DATE
        
        # Initialize dictionaries to store scalers for each ticker
        feature_scalers = {}
        target_scalers = {}
        
        # Get unique tickers
        unique_tickers = df['ticker'].unique()
        
        # Initialize lists to store scaled data for each ticker
        train_x_scaled_list = []
        test_x_scaled_list = []
        train_y_scaled_list = []
        test_y_scaled_list = []
        
        # Process each ticker separately
        for ticker in unique_tickers:
            # Get data for this ticker
            ticker_data = df[df['ticker'] == ticker]
            
            # Skip if no data for this ticker
            if len(ticker_data) == 0:
                logger.warning(f"No data found for ticker {ticker}, skipping...")
                continue
                
            # Split ticker data into train and test
            ticker_train = ticker_data[train_mask]
            ticker_test = ticker_data[test_mask]
            
            # Skip if no training data
            if len(ticker_train) == 0:
                logger.warning(f"No training data found for ticker {ticker}, skipping...")
                continue
                
            # Create scalers for this ticker
            feature_scalers[ticker] = StandardScaler()
            target_scalers[ticker] = StandardScaler()
            
            # Get features and targets for this ticker
            ticker_train_features = ticker_train[self.config.FEATURES]
            ticker_train_targets = ticker_train[self.config.DEPENDENT_VARIABLES]
            
            # Scale training data
            ticker_train_x = feature_scalers[ticker].fit_transform(ticker_train_features)
            ticker_train_y = target_scalers[ticker].fit_transform(ticker_train_targets)
            
            # Get hot encoding columns
            ticker_hot_encoding = ticker_data[ticker_dummies.columns]
            other_hot_encoding = ticker_data[self.config.HOT_ENCODING_FEATURES]
            
            # Combine scaled features with hot encoding for training
            ticker_train_x = np.hstack([
                ticker_train_x,
                ticker_hot_encoding[train_mask].values,
                other_hot_encoding[train_mask].values
            ])
            
            # Process test data if available
            if len(ticker_test) > 0:
                ticker_test_features = ticker_test[self.config.FEATURES]
                ticker_test_targets = ticker_test[self.config.DEPENDENT_VARIABLES]
                
                ticker_test_x = feature_scalers[ticker].transform(ticker_test_features)
                ticker_test_y = target_scalers[ticker].transform(ticker_test_targets)
                
                # Combine scaled features with hot encoding for testing
                ticker_test_x = np.hstack([
                    ticker_test_x,
                    ticker_hot_encoding[test_mask].values,
                    other_hot_encoding[test_mask].values
                ])
                
                # Append test data
                test_x_scaled_list.append(ticker_test_x)
                test_y_scaled_list.append(ticker_test_y)
            
            # Append training data
            train_x_scaled_list.append(ticker_train_x)
            train_y_scaled_list.append(ticker_train_y)
        
        # Check if we have any data
        if not train_x_scaled_list:
            raise ValueError("No valid training data found for any ticker")
        
        # Combine all tickers' data
        train_x_scaled = np.vstack(train_x_scaled_list)
        train_y_scaled = np.vstack(train_y_scaled_list)
        
        # Combine test data if available
        if test_x_scaled_list:
            test_x_scaled = np.vstack(test_x_scaled_list)
            test_y_scaled = np.vstack(test_y_scaled_list)
        else:
            logger.warning("No test data available for any ticker")
            test_x_scaled = np.array([])
            test_y_scaled = np.array([])
        
        # Create multi-step sequences
        train_X, train_Y = lagged_sequences(train_x_scaled, train_y_scaled, self.config.LOOKBACK_WINDOW, 0)
        
        if len(test_x_scaled) > 0:
            test_X, test_Y = lagged_sequences(test_x_scaled, test_y_scaled, self.config.LOOKBACK_WINDOW, 0)
        else:
            test_X = np.array([])
            test_Y = np.array([])
        
        logger.info(f"Train X shape: {train_X.shape}")
        logger.info(f"Train Y shape: {train_Y.shape}")
        if len(test_X) > 0:
            logger.info(f"Test X shape: {test_X.shape}")
            logger.info(f"Test Y shape: {test_Y.shape}")
        
        return train_X, test_X, train_Y, test_Y

    def prepare_tft_data(self, df: pd.DataFrame, ticker: str) -> Tuple[TimeSeriesDataSet, DataLoader, DataLoader]:
        """Prepare data for TFT model"""
        df = df.copy()
        df['time_idx'] = np.arange(len(df))
        df['group'] = '0'  # Use string representation of group number
        df['ticker'] = ticker  # Keep ticker information in a separate column
        
        # Split data
        train_data = df[df['Date'] < self.config.TRAIN_END_DATE]
        val_data = df[df['Date'] >= self.config.TRAIN_END_DATE]
        
        logger.info(f"TFT train data shape: {train_data.shape}")
        logger.info(f"TFT validation data shape: {val_data.shape}")
        
        logger.info(f"Data types in train_data: {train_data.dtypes}")
        logger.info(f"Data types in val_data: {val_data.dtypes}")
        
        logger.info(f"Training data head: {train_data.head()}")
        df['group'] = df['group'].astype('category')  # Convert group to categorical type
        
        # Create TimeSeriesDataSet with MultiNormalizer
        training = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target=self.config.DEPENDENT_VARIABLES,
            group_ids=["group"],
            min_encoder_length=7,
            max_encoder_length=7,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["group"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=self.config.FEATURES,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.config.DEPENDENT_VARIABLES,
            target_normalizer=MultiNormalizer(
                [GroupNormalizer(groups=["group"], method="standard") for _ in self.config.DEPENDENT_VARIABLES]
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
        )
        
        # Explicitly fit the normalizer if necessary
        training.target_normalizer.fit(train_data[self.config.DEPENDENT_VARIABLES], train_data)
        
        validation = TimeSeriesDataSet.from_dataset(training, val_data, stop_randomization=True)
        
        # Create dataloaders
        train_dataloader = training.to_dataloader(train=True, batch_size=self.config.BATCH_SIZE, num_workers=5)
        val_dataloader = validation.to_dataloader(train=False, batch_size=self.config.BATCH_SIZE, num_workers=5)
        
        return training, train_dataloader, val_dataloader
    
    def create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch dataloaders for the basic model"""
        logger.info(f"Creating dataloaders with shapes:")
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"X_test: {X_test.shape}")
        logger.info(f"y_test: {y_test.shape}")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader 