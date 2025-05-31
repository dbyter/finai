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
    
    # Get current ticker from the index
    current_ticker = original_sequence.index.get_level_values('ticker')[0]
    
    # Create dummy variables for all tickers
    ticker_dummies = np.zeros((lookback_window, len(config.TICKERS)))
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
            benchmark_total = benchmark_cumulative.iloc[-1]
            if isinstance(benchmark_total, pd.Series):
                benchmark_total = benchmark_total.iloc[0]
            metrics['benchmark_return'] = float(benchmark_total)
            metrics['excess_return'] = float(total_return - benchmark_total)
            
            # Information ratio
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            benchmark_annualized = (1 + benchmark_total) ** (1/years) - 1
            information_ratio = (annualized_return - benchmark_annualized) / tracking_error
            metrics['information_ratio'] = float(information_ratio)
            
            # Calculate benchmark Sharpe ratio
            benchmark_excess_returns = benchmark_returns - 0.02/252
            benchmark_sharpe = np.sqrt(252) * benchmark_excess_returns.mean() / benchmark_returns.std()
            metrics['benchmark_sharpe_ratio'] = float(benchmark_sharpe)
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {str(e)}")
            logger.warning(f"Benchmark returns shape: {benchmark_returns.shape}")
            logger.warning(f"Benchmark returns head: {benchmark_returns.head()}")
    
    return metrics

def optimize_portfolio(predicted_returns, predicted_stddevs, risk_aversion=1.0):
    """Optimize portfolio weights using mean-variance optimization with predicted standard deviations"""
    n_assets = len(predicted_returns)
    
    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        # Use predicted standard deviations to calculate portfolio variance
        portfolio_variance = np.sum((predicted_stddevs * weights)**2)
        # Minimize negative return plus risk-adjusted variance
        # (scipy minimizes, so negative return maximizes return)
        return -portfolio_return + risk_aversion * portfolio_variance
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # weights sum to 1
    ]
    
    # Set bounds: minimum 0%, maximum 20% per asset
    bounds = tuple((0, 0.20) for _ in range(n_assets))
    
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
    test_data = data.loc[test_start:test_end]
    logger.info(f"Test period: {test_data.index.get_level_values('Date')[0]} to {test_data.index.get_level_values('Date')[-1]}")
    
    # Filter out tickers that don't have data for the entire test period
    available_dates = test_data.index.get_level_values('Date').unique()
    ticker_counts = test_data.groupby(level='ticker').size()
    logger.info(f"Ticker counts: {ticker_counts.to_dict()}")
    valid_tickers = ticker_counts[ticker_counts == len(available_dates)].index.tolist()
    
    # Also filter for tickers that have scalers
    valid_tickers = [t for t in valid_tickers if t in feature_scalers and t in target_scalers]
    
    logger.info(f"Found {len(valid_tickers)} tickers with complete data out of {len(config.TICKERS)} total tickers")
    
    # Filter test_data to only include valid tickers
    test_data = test_data[test_data.index.get_level_values('ticker').isin(valid_tickers)]
    
    # Get benchmark data
    logger.info("Downloading benchmark data...")
    benchmark = yf.download('^GSPC', 
                          start=test_data.index.get_level_values('Date')[0], 
                          end=test_data.index.get_level_values('Date')[-1],
                          progress=False)
    benchmark = benchmark.reset_index()
    benchmark.columns = benchmark.columns.droplevel('Ticker')
    benchmark_daily_returns = benchmark['Close'].pct_change()
    
    # Get unique dates for iteration
    unique_dates = test_data.index.get_level_values('Date').unique()
    
    # Initialize portfolio tracking
    portfolio_returns = pd.Series(index=unique_dates, dtype=float)
    equal_weight_returns = pd.Series(index=unique_dates, dtype=float)
    portfolio_weights = pd.DataFrame(index=unique_dates, columns=valid_tickers)
    
    # Get all Mondays in the test period
    mondays = pd.date_range(start=unique_dates[0], end=unique_dates[-1], freq='W-MON')
    
    # Initialize with equal weights for the first day
    if len(unique_dates) > 0:
        portfolio_weights.loc[unique_dates[0]] = 1.0 / len(valid_tickers)
    
    # Track predictions for each horizon
    predictions = {
        '1d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '7d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '14d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '21d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '28d': pd.DataFrame(index=unique_dates, columns=valid_tickers)
    }
    
    # Track standard deviation predictions
    stddev_predictions = {
        '7d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '14d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '21d': pd.DataFrame(index=unique_dates, columns=valid_tickers),
        '28d': pd.DataFrame(index=unique_dates, columns=valid_tickers)
    }
    
    # Run backtest
    for i, current_date in enumerate(unique_dates[:-1]):  # Exclude last date as we need next day's return
        logger.info(f"Processing date: {current_date}")
        
        # Process each ticker
        ticker_predictions = {}
        ticker_stddevs = {}
        for ticker in valid_tickers:
            try:
                # Get ticker data up to current date
                ticker_data = test_data.loc[pd.IndexSlice[:current_date, ticker], :]
                
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
                        predictions_scaled = model(X_tensor).cpu().numpy()
                        
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
            
            # Optimize and set new weights
            weights = optimize_portfolio(pred_returns, pred_stddevs)
            # Set weights for tickers with predictions
            for ticker, weight in zip(ticker_predictions.keys(), weights):
                portfolio_weights.loc[current_date, ticker] = weight
            logger.info(f"Monday {current_date}: Set new weights for {len(ticker_predictions)} tickers")
        else:
            # Carry forward the last set of weights
            prev_dates = portfolio_weights.index[portfolio_weights.index < current_date]
            if len(prev_dates) == 0:
                # If this is the first date, initialize with equal weights
                portfolio_weights.loc[current_date] = 1.0 / len(valid_tickers)
                logger.info(f"First date {current_date}: Initialized with equal weights")
            else:
                prev_date = prev_dates[-1]
                portfolio_weights.loc[current_date] = portfolio_weights.loc[prev_date]
                logger.info(f"Non-Monday {current_date}: Carried forward weights from {prev_date}")
        
        # Calculate daily portfolio return and equal-weight return
        if i > 0:
            # Get returns for current date for all tickers
            current_returns = test_data.loc[pd.IndexSlice[current_date, :], 'Return']
            current_returns = current_returns.reset_index(level='Date', drop=True)
            
            # Calculate optimized portfolio return
            current_weights = portfolio_weights.loc[current_date]
            daily_return = np.sum(current_weights * current_returns)
            portfolio_returns.loc[current_date] = daily_return
            
            # Calculate equal-weight portfolio return
            equal_weight = 1.0 / len(valid_tickers)
            equal_weight_return = np.sum(current_returns * equal_weight)
            equal_weight_returns.loc[current_date] = equal_weight_return
            
            logger.info(f"Date {current_date}: Optimized return = {daily_return:.4f}, Equal-weight return = {equal_weight_return:.4f}")
    
    # Calculate performance metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio_returns, benchmark_daily_returns)
    equal_weight_metrics = calculate_portfolio_metrics(equal_weight_returns, benchmark_daily_returns)
    
    # Log results
    logger.info("\nOptimized Portfolio Performance Metrics:")
    logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {portfolio_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    
    logger.info("\nEqual-Weight Portfolio Performance Metrics:")
    logger.info(f"Total Return: {equal_weight_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {equal_weight_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {equal_weight_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {equal_weight_metrics['sharpe_ratio']:.2f}")
    
    # Log benchmark metrics if they exist
    if 'benchmark_sharpe_ratio' in portfolio_metrics:
        logger.info(f"\nS&P 500 Sharpe Ratio: {portfolio_metrics['benchmark_sharpe_ratio']:.2f}")
    if all(key in portfolio_metrics for key in ['excess_return', 'information_ratio']):
        logger.info(f"Excess Return vs S&P 500: {portfolio_metrics['excess_return']:.2%}")
        logger.info(f"Information Ratio: {portfolio_metrics['information_ratio']:.2f}")
    elif 'excess_return' in portfolio_metrics:
        logger.info(f"Excess Return vs S&P 500: {portfolio_metrics['excess_return']:.2%}")
    
    # Calculate ticker-specific metrics
    ticker_metrics = {}
    for ticker in valid_tickers:
        # Calculate average weight
        avg_weight = portfolio_weights[ticker].mean()
        
        # Calculate ticker returns
        ticker_returns = test_data.loc[pd.IndexSlice[:, ticker], 'Return']
        ticker_returns.index = ticker_returns.index.get_level_values('Date')
        
        # Calculate annualized return
        total_return = (1 + ticker_returns).cumprod().iloc[-1] - 1
        years = len(ticker_returns) / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        ticker_metrics[ticker] = {
            'average_weight': avg_weight,
            'annualized_return': annualized_return
        }
    
    # Save ticker metrics to CSV
    ticker_metrics_df = pd.DataFrame.from_dict(ticker_metrics, orient='index')
    ticker_metrics_df.index.name = 'ticker'
    ticker_metrics_df.to_csv('ticker_metrics.csv', float_format='%.4f')
    
    return {
        'portfolio_returns': portfolio_returns,
        'equal_weight_returns': equal_weight_returns,
        'portfolio_weights': portfolio_weights,
        'predictions': predictions,
        'stddev_predictions': stddev_predictions,
        'metrics': {
            'portfolio': portfolio_metrics,
            'equal_weight': equal_weight_metrics
        },
        'ticker_metrics': ticker_metrics,
        'valid_tickers': valid_tickers
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
    data = data.sort_values(['Date', 'ticker']).groupby(['Date', 'ticker']).last()
    
    # No need to reset_index and set_index again since groupby already creates the multi-index
    logger.info(f"Data shape after handling ticker-date duplicates: {data.shape}")
    logger.info(f"Index levels: {data.index.names}")
    
    # Run backtest
    results = backtest_portfolio(model, feature_scalers, target_scalers, data, config, device)
    
    # Save results
    results['portfolio_returns'].to_csv('portfolio_returns.csv')
    results['portfolio_weights'].to_csv('portfolio_weights.csv')
    for horizon, preds in results['predictions'].items():
        preds.to_csv(f'predictions_{horizon}.csv') 