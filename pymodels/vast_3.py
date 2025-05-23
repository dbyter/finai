import torch
import logging
from .data_model import DataModel
from .config import ModelConfig
from .data_processor import DataProcessor
from .models import  ModelTrainer, TransformerModel, LinearBaseline
from .utils import plot_predictions, plot_training_losses
import traceback
import torch.nn.functional as F          # <- add this line

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info("Loading data from cache")
    d = DataModel(use_cache=True)
    all_data = d.get_data()
    all_data['AAPL'].to_csv('test.csv')
    # combined_df = pd.concat(all_data.values(), ignore_index=True)
    
    # Filter data for specified tickers
    all_data = {k: v for k, v in all_data.items() if k in config.TICKERS}
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    # logger.info(f"Filtered tickers: {list(all_data.keys())}")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    

    # Prepare data
    df = data_processor.prepare_data(combined_df)
    
    # Create hot encoding features
    # hotencode_features = [f'hotencode_ticker_{i}' for i in range(len(config.TICKERS))]
    # hotencode_key = config.TICKERS.index(ticker)
    
    # Prepare data for basic model
    X_train, X_test, y_train, y_test = data_processor.create_sequences_tft_with_ticker_scaling(df)
    
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error(f"No data available after processing")
        return
        
    train_loader, test_loader = data_processor.create_dataloaders(
        X_train, y_train, X_test, y_test
    )

    batch = next(iter(train_loader))
    _, y = batch
    logger.info(f"y min: {y.min().item()}, y max: {y.max().item()}, y std: {y.std().item()}")
    
    logger.info(f"\nTraining basic model...")
    basic_model = TransformerModel(config.FEATURE_COUNT, len(config.DEPENDENT_VARIABLES), 128).to(device)

    # basic_model = LinearBaseline(len(config.FEATURES), 7, len(config.DEPENDENT_VARIABLES)).to(device)

    # x_small, y_small = next(iter(train_loader))
    # x_small, y_small = x_small[:32].to(device), y_small[:32].to(device)

    # opt = torch.optim.Adam(basic_model.parameters(), lr=1e-2)  # BIG LR, no clipping
    # for step in range(300):
    #     opt.zero_grad()
    #     loss = F.mse_loss(basic_model(x_small), y_small)
    #     loss.backward()
    #     opt.step()
    #     if step % 50 == 0:
    #         print(step, loss.item())

    trainer = ModelTrainer(config, device)
    
    # Train basic model
    losses = trainer.train_basic_model(basic_model, train_loader)
    plot_training_losses(losses, config)
    
    # Evaluate basic model
    metrics = trainer.evaluate_basic_model(basic_model, test_loader)
    for var_name in config.DEPENDENT_VARIABLES:
        logger.info(f"Loss for {var_name}: {var_name} MAE: {metrics[var_name]['mae']}, RMSE: {metrics[var_name]['rmse']}")
    plot_predictions(metrics, config)
    
    # Save basic model
    # save_model_to_s3(
    #     basic_model.state_dict(),
    #     metrics,
    #     config.S3_BUCKET,
    #     f'models/basic_model_{ticker}.pth'
    # )
    

if __name__ == '__main__':
    main()
