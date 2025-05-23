import torch
import logging
from .data_model import DataModel
from .config import ModelConfig
from .data_processor import DataProcessor
from .models import StockPredictor, TFTModel, ModelTrainer
from .utils import save_model_to_s3, plot_predictions, plot_training_losses
import lightning.pytorch as pl                          # ← unified namespace
from lightning.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger
import traceback
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
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
    device = torch.device("cpu")  # Using CPU as specified
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info("Loading data from cache")
    d = DataModel(use_cache=True)
    all_data = d.get_data()
    logger.info(f"Available tickers: {list(all_data.keys())}")
    
    # Filter data for specified tickers
    all_data = {k: v for k, v in all_data.items() if k in config.TICKERS}
    logger.info(f"Filtered tickers: {list(all_data.keys())}")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Process each ticker
    for ticker, df in all_data.items():
        try:
            logger.info(f"\nProcessing {ticker}...")
            logger.info(f"Initial data shape: {df.shape}")
            
            # Prepare data
            df = data_processor.prepare_data(df)
            
            # Create hot encoding features
            hotencode_features = [f'hotencode_ticker_{i}' for i in range(len(config.TICKERS))]
            hotencode_key = config.TICKERS.index(ticker)
            
            # Prepare data for basic model
            X_train, X_test, y_train, y_test = data_processor.create_sequences(
                df, hotencode_features, hotencode_key
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                logger.error(f"No data available for {ticker} after processing")
                continue
                
            train_loader, test_loader = data_processor.create_dataloaders(
                X_train, y_train, X_test, y_test
            )
            
            # Train and evaluate basic model
            # logger.info(f"\nTraining basic model for {ticker}...")
            # basic_model = StockPredictor(config).to(device)
            # trainer = ModelTrainer(config, device)
            
            # # Train basic model
            # losses = trainer.train_basic_model(basic_model, train_loader)
            # plot_training_losses(losses, config)
            
            # # Evaluate basic model
            # metrics = trainer.evaluate_basic_model(basic_model, test_loader)
            # plot_predictions(metrics, config)
            
            # # Save basic model
            # save_model_to_s3(
            #     basic_model.state_dict(),
            #     metrics,
            #     config.S3_BUCKET,
            #     f'models/basic_model_{ticker}.pth'
            # )
            
            # Prepare and train TFT model

            logger.info(f"\nTraining TFT model for {ticker}...")
            training_dataset, train_dataloader, val_dataloader = data_processor.prepare_tft_data(df, ticker)
            
            # tft_model = TFTModel(config, training_dataset)
            # tft_trainer = pl.Trainer(
            #     max_epochs=config.NUM_EPOCHS,
            #     accelerator="cpu",
            #     enable_model_summary=True,
            #     gradient_clip_val=0.1,
            #     callbacks=[
            #         pl.callbacks.EarlyStopping(
            #             monitor='val_loss',
            #             patience=5,
            #             mode='min'
            #         ),
            #         pl.callbacks.ModelCheckpoint(
            #             dirpath=f'checkpoints/{ticker}',
            #             filename='tft-{epoch:02d}-{val_loss:.2f}',
            #             save_top_k=3,
            #             monitor='val_loss',
            #             mode='min'
            #         ),
            #         pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            #     ],
            #     logger=pl.loggers.TensorBoardLogger(
            #         save_dir=f'logs/{ticker}',
            #         name='tft_model'
            #     )
            # )
            
            # tft_trainer.fit(
            #     tft_model,
            #     train_dataloaders=train_dataloader,
            #     val_dataloaders=val_dataloader,
            # )
            
            tft_model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate         = config.TFT_LEARNING_RATE,
                hidden_size           = config.TFT_HIDDEN_SIZE,
                attention_head_size   = config.TFT_ATTENTION_HEADS,
                dropout               = config.TFT_DROPOUT,
                hidden_continuous_size= config.TFT_HIDDEN_CONTINUOUS_SIZE,
                loss                  = QuantileLoss(),        # TFT logs this for you
            )


            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5),
                ModelCheckpoint(
                    dirpath   = f"checkpoints/{ticker}",
                    filename  = "tft-{epoch:02d}-{val_loss:.2f}",
                    monitor   = "val_loss",
                    mode      = "min",
                    save_top_k= 3,
                ),
                LearningRateMonitor("epoch"),
            ]

            trainer = pl.Trainer(
                max_epochs        = config.NUM_EPOCHS,
                accelerator       = "cpu",          # or "auto" to use GPU if present
                gradient_clip_val = 0.1,
                callbacks         = callbacks,
                logger            = TensorBoardLogger(f"logs/{ticker}", name="tft"),
            )

            trainer.fit(
                tft_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # Evaluate TFT model
            logger.info(f"Evaluating TFT model for {ticker}...")
            pred_container = tft_model.predict(
                val_dataloader,
                mode="prediction",   # default
                return_y=True        # ask for ground truth
            )              
            logger.info(f"{len(pred_container)}")       
            logger.info(f"Predictions shape: {pred_container.shape}")

            y_pred = pred_container.output    # tensor (N, pred_len, n_targets)
            y_true = pred_container.y         # same shape
            # flatten the horizon so metrics are per-time-step, not per-series
            y_pred = y_pred.reshape(-1, y_pred.size(-1))   # (N * pred_len, n_targets)
            y_true = y_true.reshape(-1, y_true.size(-1))

            target_names = training_dataset.target_names   # e.g. ['close', 'volume']

            for i, name in enumerate(training_dataset.target_names):
                mse  = torch.mean((y_pred[:, i] - y_true[:, i]) ** 2).item()
                mae  = torch.mean(torch.abs(y_pred[:, i] - y_true[:, i])).item()
                rmse = mse ** 0.5
                logger.info(f"{name.upper()} —  MSE {mse:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f}")
                plot_predictions(
                    {
                        "predictions": y_pred[:, i].cpu().numpy(),
                        "actuals":     y_true[:, i].cpu().numpy(),
                        "ticker":      ticker,
                        "target_name": name,
                    },
                    config,
                )

            # Save TFT model
            save_model_to_s3(
                tft_model.state_dict(),
                {},  # TFT metrics are handled internally
                config.S3_BUCKET,
                f'models/tft_model_{ticker}.pth'
            )
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}, {traceback.format_exc()}")
            continue
    
    logger.info("Process completed successfully")

if __name__ == '__main__':
    main()
