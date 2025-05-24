import torch
import logging
from .data_model import DataModel
from .config import ModelConfig
from .data_processor import DataProcessor
from .models import  ModelTrainer, TransformerModel, LinearBaseline
from .utils import plot_predictions, plot_training_losses, save_model_to_s3
import traceback
import torch.nn.functional as F          # <- add this line
from datetime import datetime
import pandas as pd
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    # Initialize configuration
    config = ModelConfig()
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info("Loading data from cache")
    d = DataModel(use_cache=True)
    all_data = d.get_data()
    high_value_tickers =  (list(set([ticker for ticker, values in all_data.items() if values['ebitda'].mean() > 1e10])))
    print (high_value_tickers)
    useful_columns = config.FEATURES + config.DEPENDENT_VARIABLES + config.HOT_ENCODING_FEATURES + ['Date', 'total_assets', 'ticker', 'total_liabilities']
    limit_features_df = [x[useful_columns] for x in all_data.values()]
    # all_data['AAPL'].to_csv('test.csv')
    # combined_df = pd.concat(limit_features_df, ignore_index=True)
    
    # Filter data for specified tickers
    all_data = {k: v for k, v in all_data.items() if k in config.TICKERS}
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    # logger.info(f"Filtered tickers: {list(all_data.keys())}")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    df = data_processor.prepare_data(combined_df)
    
    # Prepare data for basic model
    X_train, X_test, y_train, y_test = data_processor.create_sequences_tft_with_ticker_scaling(df)
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error(f"No data available after processing")
        return
        
    train_loader, test_loader = data_processor.create_dataloaders(
        X_train, y_train, X_test, y_test
    )

    logger.info(f"\nTraining basic model...")
    # basic_model = TransformerModel(config.FEATURE_COUNT, len(config.DEPENDENT_VARIABLES), 32).to(device)
    basic_model = LinearBaseline(config.FEATURE_COUNT, config.LOOKBACK_WINDOW, len(config.DEPENDENT_VARIABLES)).to(device)

    total_params = sum(p.numel() for p in basic_model.parameters())
    trainable_params = sum(p.numel() for p in basic_model.parameters() if p.requires_grad)

    # Example usage:
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = ModelTrainer(config, device)
    
    # Train basic model
    losses = trainer.train_basic_model(basic_model, train_loader)
    # plot_training_losses(losses, config)
    
    # Evaluate basic model
    metrics = trainer.evaluate_basic_model(basic_model, test_loader, train_loader)
    for var_name in config.DEPENDENT_VARIABLES:
        logger.info(f"Loss for {var_name}:")
        logger.info(f"Train - MAE: {metrics['train'][var_name]['mae']:.4f}, RMSE: {metrics['train'][var_name]['rmse']:.4f}")
        logger.info(f"Test  - MAE: {metrics['test'][var_name]['mae']:.4f}, RMSE: {metrics['test'][var_name]['rmse']:.4f}")
    # plot_predictions(metrics, config)
    
    # Save basic model
    save_model_to_s3(
        basic_model.state_dict(),
        metrics,
        config.S3_BUCKET,
        f'models/basic_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    )
    

if __name__ == '__main__':
    main()
