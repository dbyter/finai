import torch
import pandas as pd
import numpy as np
import boto3
import io
import logging
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from pymodels.config import ModelConfig
from pymodels.models import TransformerModel
import pickle
from sklearn.preprocessing import StandardScaler
from pymodels.data_model import DataModel
from pymodels.data_processor import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_from_s3(bucket_name, file_path):
    """Load file from S3"""
    try:
        s3_client = boto3.client('s3')
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket_name, file_path, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Error loading file from S3: {str(e)}")
        raise

def load_model_and_scalers(bucket_name, model_path, scaler_path):
    """Load model and scalers from S3"""
    try:
        # Load model
        model_buffer = load_from_s3(bucket_name, model_path)
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info(f"Loading model to device: {device}")
        
        saved_dict = torch.load(
            model_buffer,
            map_location=device,
            weights_only=False
        )
        
        # Initialize model with correct parameters
        config = ModelConfig()
        logger.info(f"Model configuration - Features: {config.FEATURE_COUNT}, Dependent Variables: {len(config.DEPENDENT_VARIABLES)}")
        
        model = TransformerModel(
            n_features=config.FEATURE_COUNT,
            n_dependent_variables=len(config.DEPENDENT_VARIABLES),
            d_model=32  # This matches the value used in vast_3.py
        )
        
        # Validate model state dict
        model_state = saved_dict['model_state']
        
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        # Load scalers
        scaler_buffer = load_from_s3(bucket_name, scaler_path)
        scalers = pickle.load(scaler_buffer)
        feature_scalers = scalers['feature_scalers']
        target_scalers = scalers['target_scalers']
        
        return model, feature_scalers, target_scalers, device
    except Exception as e:
        logger.error(f"Error loading model or scalers: {str(e)}")
        raise

def create_lagged_sequence(scaled_data, original_data, lookback_window, config):
    """Create lagged sequence for prediction"""
    if len(scaled_data) < lookback_window:
        raise ValueError(f"Not enough data points. Need at least {lookback_window} points.")
    
    # Get the last lookback_window points
    scaled_sequence = scaled_data[-lookback_window:]
    original_sequence = original_data[-lookback_window:]
    
    # Create dummy variables for all tickers
    ticker_dummies = np.zeros((lookback_window, len(config.TICKERS)))
    current_ticker = original_sequence['ticker'].iloc[0]
    ticker_idx = config.TICKERS.index(current_ticker)
    ticker_dummies[:, ticker_idx] = 1
    
    # Combine scaled features with ticker dummies
    combined_features = np.hstack([scaled_sequence, ticker_dummies])
    
    # Ensure we have the correct number of features
    expected_features = config.FEATURE_COUNT
    if combined_features.shape[1] != expected_features:
        raise ValueError(f"Feature count mismatch. Expected {expected_features}, got {combined_features.shape[1]}. Base features: {scaled_sequence.shape[1]}, Ticker dummies: {ticker_dummies.shape[1]}")
    
    return combined_features.reshape(1, lookback_window, -1)  # Reshape for model input

def calculate_portfolio_metrics(returns, benchmark_returns=None):
    """Calculate portfolio performance metrics"""
    metrics = {}
    
    # Handle NaN values in returns
    returns = returns.fillna(0)
    
    # Portfolio metrics
    cumulative_return = (1 + returns).cumprod() - 1
    total_return = cumulative_return.iloc[-1]
    
    # Annualized return
    years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1/years) - 1
    
    # Maximum drawdown
    rolling_max = cumulative_return.expanding().max()
    drawdowns = cumulative_return - rolling_max
    max_drawdown = drawdowns.min()
    
    # Sharpe ratio (assuming risk-free rate of 0.02)
    excess_returns = returns - 0.02/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    metrics['total_return'] = float(total_return)  # Convert to float to avoid Series formatting issues
    metrics['annualized_return'] = float(annualized_return)
    metrics['max_drawdown'] = float(max_drawdown)
    metrics['sharpe_ratio'] = float(sharpe_ratio)
    
    # Benchmark comparison if available
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            benchmark_returns = benchmark_returns.fillna(0)  # Handle NaN values
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            benchmark_total = float(benchmark_cumulative.iloc[-1])
            metrics['benchmark_return'] = benchmark_total
            metrics['excess_return'] = float(total_return - benchmark_total)
            
            # Information ratio
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            information_ratio = (annualized_return - (1 + benchmark_total) ** (1/years) + 1) / tracking_error
            metrics['information_ratio'] = float(information_ratio)
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {str(e)}")
    
    return metrics

def optimize_portfolio(predicted_returns, predicted_stddevs, risk_aversion=1.0):
    """Optimize portfolio weights using mean-variance optimization with predicted standard deviations"""
    n_assets = len(predicted_returns)
    
    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        # Use predicted standard deviations to calculate portfolio variance
        portfolio_variance = np.sum((predicted_stddevs * weights)**2)
        return -(portfolio_return - risk_aversion * portfolio_variance)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(objective, initial_weights, method='SLSQP',
                     constraints=constraints, bounds=bounds)
    
    return result.x if result.success else initial_weights

def backtest_portfolio(model, feature_scalers, target_scalers, data, config, device):
    """Backtest portfolio strategy with multiple prediction horizons"""
    logger.info("Starting portfolio backtest...")
    
    # Convert test dates to datetime if they're strings
    test_start = pd.to_datetime(config.TEST_START_DATE)
    test_end = pd.to_datetime(config.TEST_END_DATE)
    
    # Get test period data
    test_data = data[test_start:test_end]
    logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Get benchmark data
    benchmark_returns = None
    try:
        logger.info("Downloading benchmark data...")
        benchmark = yf.download('AAPL', 
                              start=test_data.index[0], 
                              end=test_data.index[-1],
                              progress=False)
        if not benchmark.empty:
            benchmark_returns = benchmark['Close'].pct_change()
            logger.info(f"Successfully downloaded benchmark data with {len(benchmark_returns)} points")
        else:
            logger.warning("Downloaded benchmark data is empty")
    except Exception as e:
        logger.warning(f"Failed to download benchmark data: {str(e)}")
    
    # Initialize portfolio tracking
    portfolio_returns = pd.Series(index=test_data.index, dtype=float)
    portfolio_weights = pd.DataFrame(index=test_data.index, columns=config.TICKERS)
    
    # Get all Mondays in the test period
    mondays = pd.date_range(start=test_data.index[0], end=test_data.index[-1], freq='W-MON')
    
    # Track predictions for each horizon
    predictions = {
        '1d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '7d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '14d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '21d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '28d': pd.DataFrame(index=test_data.index, columns=config.TICKERS)
    }
    
    # Track standard deviation predictions
    stddev_predictions = {
        '7d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '14d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '21d': pd.DataFrame(index=test_data.index, columns=config.TICKERS),
        '28d': pd.DataFrame(index=test_data.index, columns=config.TICKERS)
    }
    
    # Run backtest
    for i in range(len(test_data) - 1):
        current_date = test_data.index[i]
        
        # Process each ticker
        ticker_predictions = {}
        ticker_stddevs = {}
        for ticker in config.TICKERS:
            try:
                # Get ticker data up to current date
                ticker_mask = test_data['ticker'] == ticker
                ticker_data = test_data[ticker_mask].iloc[:i+1]
                
                if len(ticker_data) < config.LOOKBACK_WINDOW:
                    continue
                
                # Scale features using ticker-specific scaler
                features = ticker_data[config.FEATURES]
                features_scaled = feature_scalers[ticker].transform(features)
                
                # Create lagged sequence
                sequence = create_lagged_sequence(features_scaled, ticker_data, config.LOOKBACK_WINDOW, config)
                
                # Get predictions
                with torch.no_grad():
                    try:
                        X_tensor = torch.FloatTensor(sequence).to(device)
                        logger.debug(f"Input tensor shape: {X_tensor.shape}")
                        predictions_scaled = model(X_tensor).cpu().numpy()
                        logger.debug(f"Output tensor shape: {predictions_scaled.shape}")
                        
                        # Unscale predictions using ticker-specific scaler
                        predictions_unscaled = target_scalers[ticker].inverse_transform(predictions_scaled)
                        
                        # Store predictions for each horizon
                        ticker_predictions[ticker] = {
                            '1d': predictions_unscaled[0, 0],   # Next_Day_Return
                            '7d': predictions_unscaled[0, 1],   # Next_7_Day_Return
                            '14d': predictions_unscaled[0, 2],  # Next_14_Day_Return
                            '21d': predictions_unscaled[0, 3],  # Next_21_Day_Return
                            '28d': predictions_unscaled[0, 4]   # Next_28_Day_Return
                        }
                        
                        # Store standard deviation predictions
                        ticker_stddevs[ticker] = {
                            '7d': predictions_unscaled[0, 5],   # Next_7_Day_Return_StdDev
                            '14d': predictions_unscaled[0, 6],  # Next_14_Day_Return_StdDev
                            '21d': predictions_unscaled[0, 7],  # Next_21_Day_Return_StdDev
                            '28d': predictions_unscaled[0, 8]   # Next_28_Day_Return_StdDev
                        }
                    except Exception as e:
                        logger.error(f"Error in model prediction for {ticker} on {current_date}: {str(e)}")
                        logger.error(f"Input sequence shape: {sequence.shape}")
                        logger.error(f"Model device: {device}")
                        continue
            except Exception as e:
                logger.warning(f"Error processing ticker {ticker} on {current_date}: {str(e)}")
                continue
        
        # Store predictions
        for horizon in ['1d', '7d', '14d', '21d', '28d']:
            for ticker, preds in ticker_predictions.items():
                predictions[horizon].loc[current_date, ticker] = preds[horizon]
                if horizon != '1d':  # Only store stddev predictions for 7d, 14d, 21d, and 28d
                    stddev_predictions[horizon].loc[current_date, ticker] = ticker_stddevs[ticker][horizon]
        
        # Rebalance on Mondays
        if current_date in mondays and ticker_predictions:
            # Use 7-day predictions for portfolio optimization
            pred_returns = np.array([preds['7d'] for preds in ticker_predictions.values()])
            pred_stddevs = np.array([stddevs['7d'] for stddevs in ticker_stddevs.values()])
            
            # Optimize weights using predicted returns and standard deviations
            weights = optimize_portfolio(pred_returns, pred_stddevs)
            portfolio_weights.loc[current_date, list(ticker_predictions.keys())] = weights
        
        # Calculate daily portfolio return
        if i > 0:
            prev_weights = portfolio_weights.loc[portfolio_weights.index[portfolio_weights.index <= current_date][-1]]
            # Get returns for current date for all tickers
            current_returns = test_data[test_data.index == current_date].set_index('ticker')['Return']
            # Calculate portfolio return
            portfolio_returns.loc[current_date] = np.sum(prev_weights * current_returns)
    
    # Calculate performance metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio_returns, benchmark_returns)
    
    # Log results
    logger.info("\nPortfolio Performance Metrics:")
    logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {portfolio_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    
    # Log benchmark metrics if they exist
    if all(key in portfolio_metrics for key in ['excess_return', 'information_ratio']):
        logger.info(f"Excess Return vs Benchmark: {portfolio_metrics['excess_return']:.2%}")
        logger.info(f"Information Ratio: {portfolio_metrics['information_ratio']:.2f}")
    elif 'excess_return' in portfolio_metrics:
        logger.info(f"Excess Return vs Benchmark: {portfolio_metrics['excess_return']:.2%}")
    
    return {
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'portfolio_weights': portfolio_weights,
        'predictions': predictions,
        'stddev_predictions': stddev_predictions,
        'metrics': portfolio_metrics
    }

if __name__ == "__main__":
    # Load configuration
    config = ModelConfig()
    
    # Define paths
    model_path = 'models/basic_model_20250524_172029.pth'
    scaler_path = 'scalers/scalers_20250524_172051.pkl'
    
    # Load model and scalers
    model, feature_scalers, target_scalers, device = load_model_and_scalers(config.S3_BUCKET, model_path, scaler_path)
    
    # Load data
    data_model = DataModel(use_cache=True)
    all_data = data_model.get_data()
    
    # Filter data for specified tickers
    all_data = {k: v for k, v in all_data.items() if k in config.TICKERS}
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    
    # Initialize data processor and prepare data
    data_processor = DataProcessor(config)
    data = data_processor.prepare_data(combined_df)
    
    # After preparation, ensure Date is datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Handle duplicate timestamps by keeping the last entry for each ticker-date combination
    data = data.sort_values(['Date', 'ticker']).groupby(['Date', 'ticker']).last().reset_index()
    data.set_index('Date', inplace=True)
    
    logger.info(f"Data shape after handling ticker-date duplicates: {data.shape}")
    # Run backtest
    results = backtest_portfolio(model, feature_scalers, target_scalers, data, config, device)
    
    # Save results
    results['portfolio_returns'].to_csv('portfolio_returns.csv')
    results['portfolio_weights'].to_csv('portfolio_weights.csv')
    for horizon, preds in results['predictions'].items():
        preds.to_csv(f'predictions_{horizon}.csv') 