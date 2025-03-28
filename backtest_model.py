import torch
import pandas as pd
import numpy as np
import boto3
import io
import logging
from torch_play import StockPredictor, preprocess_data
import yfinance as yf

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
        saved_dict = torch.load(buffer)
        
        model = StockPredictor()
        model.load_state_dict(saved_dict['model_state'])
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
    
    # Preprocess data
    X, _ = preprocess_data(data)
    
    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # Create trading signals (1 for long, 0 for no position)
    signals = (predictions > 0).astype(int)
    
    # Calculate strategy returns
    data['Strategy_Position'] = signals
    data['Strategy_Returns'] = data['Return'] * data['Strategy_Position'].shift(1)
    
    # Get benchmark data
    benchmark = yf.download(benchmark_ticker, 
                          start=data.index[0], 
                          end=data.index[-1])
    benchmark['Return'] = benchmark['Adj Close'].pct_change()
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(data['Strategy_Returns'].dropna())
    benchmark_metrics = calculate_metrics(benchmark['Return'].dropna())
    
    # Log results
    logger.info("\nStrategy Performance Metrics:")
    logger.info(f"Total Return: {strategy_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {strategy_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {strategy_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {strategy_metrics['win_rate']:.2%}")
    
    logger.info("\nBenchmark (S&P 500) Performance Metrics:")
    logger.info(f"Total Return: {benchmark_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {benchmark_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {benchmark_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {benchmark_metrics['win_rate']:.2%}")
    

    
    return {
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'strategy_returns': data['Strategy_Returns'],
        'benchmark_returns': benchmark['Return']
    }

if __name__ == "__main__":
    # Load model from S3
    bucket_name = 'tradingmodelsahmed'
    model_path = 'models/model_with_metrics.pth'
    model = load_model(bucket_name, model_path)
    
    # Load test data (you might want to modify this part based on your data structure)
    from torch_play import load_data, TICKERS, DATA_FOLDER
    test_data = load_data(TICKERS, DATA_FOLDER)
    
    # Run backtest for each ticker
    all_results = {}
    for ticker in TICKERS:
        logger.info(f"\nBacktesting {ticker}...")
        results = backtest_strategy(model, test_data[ticker])
        all_results[ticker] = results
    
    # Aggregate results across all tickers
    all_strategy_returns = pd.concat([res['strategy_returns'] for res in all_results.values()])
    portfolio_metrics = calculate_metrics(all_strategy_returns)
    
    logger.info("\nPortfolio Performance Metrics (Equal Weight):")
    logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {portfolio_metrics['annualized_return']:.2%}")
    logger.info(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {portfolio_metrics['win_rate']:.2%}") 