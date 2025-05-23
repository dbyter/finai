import torch
import pandas as pd
import numpy as np
import boto3
import io
import logging
from torch_play import StockPredictor, preprocess_data
import yfinance as yf
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_model(bucket_name, model_path):
    """Load model from S3"""
    try:
        s3_client = boto3.client('s3')
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket_name, model_path, buffer)
        buffer.seek(0)
        
        # Determine the device
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info(f"Loading model to device: {device}")
        
        # Load the model with appropriate device mapping
        saved_dict = torch.load(
            buffer,
            map_location=device,
            weights_only=False
        )
        
        model = StockPredictor()
        model.load_state_dict(saved_dict['model_state'])
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def calculate_metrics(returns):
    """Calculate trading metrics"""
    cumulative_return = (1 + returns).cumprod() - 1
    total_return = cumulative_return.iloc[-1]
    
    # Annualized return
    years = len(returns) / 252  # Trading days in a year
    annualized_return = (1 + total_return) ** (1/years) - 1
    
    # Maximum drawdown
    rolling_max = cumulative_return.expanding().max()
    drawdowns = cumulative_return - rolling_max
    max_drawdown = drawdowns.min()
    
    # Sharpe ratio (assuming risk-free rate of 0.02)
    excess_returns = returns - 0.02/252  # Daily risk-free rate
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate
    }

def backtest_strategy(model, data, benchmark_ticker='^GSPC'):
    """Backtest the trading strategy"""
    logger.info("Starting backtest...")
    
    # Split data into train/test using the same split as training
    train_size = int(0.9 * len(data))
    test_data = data.iloc[train_size:]
    logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Preprocess test data
    X, _ = preprocess_data(test_data)
    
    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
        predicted_returns = predictions[:, 0]  # First output is return
        predicted_stddev = predictions[:, 1]   # Second output is standard deviation
    
    # Create trading signals (1 for long, 0 for no position)
    signals = (predicted_returns > 0).astype(int)
    
    # Store both predictions in the preprocessed data
    preprocessed_data = test_data.copy()
    preprocessed_data = preprocessed_data.dropna()
    
    # Ensure the lengths match
    if len(preprocessed_data) > len(signals):
        preprocessed_data = preprocessed_data.iloc[len(preprocessed_data)-len(signals):]
    
    # Add predictions to preprocessed_data
    preprocessed_data['Predicted_Return'] = predicted_returns
    preprocessed_data['Predicted_StdDev'] = predicted_stddev
    preprocessed_data['Strategy_Position'] = signals
    preprocessed_data['Strategy_Returns'] = preprocessed_data['daily_return'] * preprocessed_data['Strategy_Position'].shift(1)
    
    # Log data info for debugging
    logger.info(f"Test data shape: {preprocessed_data.shape}")
    logger.info(f"Signals shape: {signals.shape}")
    
    try:
        # Get benchmark data for the test period only
        benchmark = yf.download(benchmark_ticker, 
                              start=preprocessed_data.index[0], 
                              end=preprocessed_data.index[-1],
                              progress=False)
        
        if len(benchmark) == 0:
            logger.warning(f"Failed to download benchmark data for {benchmark_ticker}. Using strategy metrics only.")
            benchmark_metrics = None
        else:
            benchmark['Return'] = benchmark['Adj Close'].pct_change()
            benchmark_metrics = calculate_metrics(benchmark['Return'].dropna())
            
            # Log benchmark results
            logger.info("\nBenchmark (SPY) Performance Metrics (Test Period):")
            logger.info(f"Total Return: {benchmark_metrics['total_return']:.2%}")
            logger.info(f"Annualized Return: {benchmark_metrics['annualized_return']:.2%}")
            logger.info(f"Maximum Drawdown: {benchmark_metrics['max_drawdown']:.2%}")
            logger.info(f"Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
            logger.info(f"Win Rate: {benchmark_metrics['win_rate']:.2%}")
    
    except Exception as e:
        logger.error(f"Error downloading benchmark data: {str(e)}")
        benchmark_metrics = None
        benchmark = pd.DataFrame()
    
    # Calculate strategy metrics
    strategy_metrics = calculate_metrics(preprocessed_data['Strategy_Returns'].dropna())
    
    # Log strategy results
    logger.info("\nStrategy Performance Metrics (Test Period):")
    logger.info(f"Total Return: {strategy_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {strategy_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {strategy_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {strategy_metrics['win_rate']:.2%}")
    
    # Add trading statistics
    logger.info("\nTrading Statistics (Test Period):")
    logger.info(f"Total Trading Days: {len(preprocessed_data)}")
    logger.info(f"Days In Market: {(preprocessed_data['Strategy_Position'] == 1).sum()}")
    logger.info(f"Percentage Time In Market: {(preprocessed_data['Strategy_Position'] == 1).mean():.2%}")
    
    return {
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'strategy_returns': preprocessed_data['Strategy_Returns'],
        'benchmark_returns': benchmark['Return'] if len(benchmark) > 0 else pd.Series(),
        'positions': preprocessed_data['Strategy_Position'],
        'test_period': (preprocessed_data.index[0], preprocessed_data.index[-1]),
        'predictions': preprocessed_data[['Predicted_Return', 'Predicted_StdDev']]  # Add predictions to return dict
    }

# Add a new function for portfolio optimization
def optimize_portfolio(predicted_returns, predicted_variance, risk_aversion=1.0):
    """
    Optimize portfolio weights using mean-variance optimization
    """
    n_assets = len(predicted_returns)
    
    def objective(weights):
        portfolio_return = np.sum(predicted_returns * weights)
        portfolio_variance = np.sum(predicted_variance * weights**2)  # Simplified variance calculation
        return -(portfolio_return - risk_aversion * portfolio_variance)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # weights sum to 1
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only constraints
    
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(objective, initial_weights, method='SLSQP',
                     constraints=constraints, bounds=bounds)
    
    return result.x if result.success else initial_weights

if __name__ == "__main__":
    # Load model from S3
    bucket_name = 'tradingmodelsahmed'
    model_path = 'models/model_with_metrics.pth'
    model = load_model(bucket_name, model_path)
    
    # Load test data
    from torch_play import load_data, TICKERS, DATA_FOLDER
    test_data = load_data(TICKERS, DATA_FOLDER)
    
    # Initialize results containers
    results_data = []
    all_results = {}
    all_predictions = {}  # Store all predictions by ticker
    
    # Run backtest for each ticker
    for ticker in TICKERS:
        logger.info(f"Backtesting {ticker}...")
        results = backtest_strategy(model, test_data[ticker])
        all_results[ticker] = results
        
        # Store predictions DataFrame with dates
        all_predictions[ticker] = results['predictions']
        
        # Calculate days held and other metrics for individual ticker results
        days_held = (results['positions'] == 1).sum()
        days_held_pct = days_held / len(results['positions'])
        
        ticker_results = {
            'Ticker': ticker,
            'Start Date': results['test_period'][0].strftime('%Y-%m-%d'),
            'End Date': results['test_period'][1].strftime('%Y-%m-%d'),
            'Days Held': f"{days_held} ({days_held_pct:.1%})",
            'Total Return': f"{results['strategy_metrics']['total_return']:.2%}",
            'Annual Return': f"{results['strategy_metrics']['annualized_return']:.2%}",
            'Max Drawdown': f"{results['strategy_metrics']['max_drawdown']:.2%}",
            'Sharpe Ratio': f"{results['strategy_metrics']['sharpe_ratio']:.2f}",
            'Win Rate': f"{results['strategy_metrics']['win_rate']:.2%}",
            'SPY Return': f"{results['benchmark_metrics']['total_return']:.2%}" if results['benchmark_metrics'] else "N/A"
        }
        results_data.append(ticker_results)

    # Create and display individual stocks table
    results_df = pd.DataFrame(results_data)
    results_df = results_df.set_index('Ticker')
    
    # Reorder columns and display individual results
    column_order = ['Start Date', 'End Date', 'Days Held', 'Total Return', 'Annual Return', 
                   'Max Drawdown', 'Sharpe Ratio', 'Win Rate', 'SPY Return']
    results_df = results_df[column_order]
    logger.info("\nIndividual Stock Performance:")
    logger.info("\n" + str(results_df))
    
    # Align all predictions and returns on common dates
    strategy_returns_list = []
    for ticker, res in all_results.items():
        returns = res['strategy_returns'].to_frame()
        returns.columns = [ticker]
        strategy_returns_list.append(returns)
    
    if strategy_returns_list:
        # Combine all returns and align on dates
        all_strategy_returns = pd.concat(strategy_returns_list, axis=1)
        all_strategy_returns = all_strategy_returns.dropna()  # Remove dates where any ticker is missing
        
        # Create aligned predictions DataFrames
        returns_pred = pd.DataFrame(index=all_strategy_returns.index)
        stddev_pred = pd.DataFrame(index=all_strategy_returns.index)
        
        for ticker in TICKERS:
            pred_df = all_predictions[ticker]
            returns_pred[ticker] = pred_df['Predicted_Return']
            stddev_pred[ticker] = pred_df['Predicted_StdDev']
        
        # Align predictions with returns
        common_dates = all_strategy_returns.index.intersection(returns_pred.index)
        all_strategy_returns = all_strategy_returns.loc[common_dates]
        returns_pred = returns_pred.loc[common_dates]
        stddev_pred = stddev_pred.loc[common_dates]
        
        # Calculate equal-weighted portfolio returns
        equal_weight_returns = all_strategy_returns.mean(axis=1)
        equal_weight_metrics = calculate_metrics(equal_weight_returns)
        
        # Calculate optimized portfolio returns
        optimized_returns = []
        optimized_weights = {}
        
        for date in common_dates:
            pred_returns = returns_pred.loc[date].values
            pred_variance = stddev_pred.loc[date].values ** 2
            
            # Optimize weights for this day
            weights = optimize_portfolio(pred_returns, pred_variance)
            optimized_weights[date] = dict(zip(TICKERS, weights))
            
            # Calculate portfolio return for this day
            daily_returns = all_strategy_returns.loc[date]
            portfolio_return = np.sum(weights * daily_returns)
            optimized_returns.append(portfolio_return)
        
        # Create optimized returns series
        optimized_returns_series = pd.Series(optimized_returns, index=common_dates)
        optimized_metrics = calculate_metrics(optimized_returns_series)
        
        # Create comparison table
        portfolio_data = {
            'Metric': [
                'Total Return',
                'Annual Return',
                'Max Drawdown',
                'Sharpe Ratio',
                'Win Rate'
            ],
            'Equal-Weight': [
                f"{equal_weight_metrics['total_return']:.2%}",
                f"{equal_weight_metrics['annualized_return']:.2%}",
                f"{equal_weight_metrics['max_drawdown']:.2%}",
                f"{equal_weight_metrics['sharpe_ratio']:.2f}",
                f"{equal_weight_metrics['win_rate']:.2%}"
            ],
            'Optimized': [
                f"{optimized_metrics['total_return']:.2%}",
                f"{optimized_metrics['annualized_return']:.2%}",
                f"{optimized_metrics['max_drawdown']:.2%}",
                f"{optimized_metrics['sharpe_ratio']:.2f}",
                f"{optimized_metrics['win_rate']:.2%}"
            ]
        }
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df = portfolio_df.set_index('Metric')
        
        logger.info("\nPortfolio Performance Comparison:")
        logger.info("\n" + str(portfolio_df))
        
        # Log average portfolio turnover for optimized strategy
        daily_turnover = []
        prev_weights = None
        for date in common_dates:
            if prev_weights is not None:
                turnover = sum(abs(optimized_weights[date][t] - prev_weights[t]) for t in TICKERS)
                daily_turnover.append(turnover)
            prev_weights = optimized_weights[date]
        
        avg_turnover = np.mean(daily_turnover) if daily_turnover else 0
        logger.info(f"\nOptimized Strategy Average Daily Turnover: {avg_turnover:.2%}")
        
        # Create weights DataFrame
        weights_df = pd.DataFrame(optimized_weights).T  # Transpose to get dates as columns
        weights_df.index = pd.to_datetime(weights_df.index)  # Ensure index is datetime
        weights_df = weights_df.sort_index()  # Sort by date
        
        # Save to CSV
        weights_file = 'portfolio_weights.csv'
        weights_df.to_csv(weights_file)
        logger.info(f"\nPortfolio weights saved to {weights_file}")
        
        # Calculate and display average weights per ticker
        avg_weights = weights_df.mean()
        logger.info("\nAverage Portfolio Weights:")
        for ticker, weight in avg_weights.sort_values(ascending=False).items():
            logger.info(f"{ticker}: {weight:.2%}")
    else:
        logger.warning("No strategy returns available to calculate portfolio metrics") 