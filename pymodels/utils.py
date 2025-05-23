import boto3
import io
import torch
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_model_to_s3(model_state: Dict[str, torch.Tensor], metrics: Dict[str, Any], 
                    bucket_name: str, s3_key: str) -> None:
    """Save model and metrics to S3"""
    try:
        save_dict = {
            'model_state': model_state,
            'test_metrics': metrics
        }
        
        buffer = io.BytesIO()
        torch.save(save_dict, buffer)
        buffer.seek(0)
        
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        logger.info(f"Model and metrics successfully saved to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Error saving model to S3: {str(e)}")
        raise

def plot_predictions(predictions: Dict[str, Dict], config) -> None:
    """Plot predictions vs actuals for each dependent variable"""
    for var_name in config.DEPENDENT_VARIABLES:
        plt.figure(figsize=(10, 6))
        plt.plot(predictions[var_name]['predictions'], label='Predictions')
        plt.plot(predictions[var_name]['actuals'], label='Actuals')
        plt.title(f'{var_name} - Predictions vs Actuals')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        # Log example predictions
        logger.info(f"\nExample {var_name} values:")
        logger.info(f"Predictions: {predictions[var_name]['predictions'][:5]}")
        logger.info(f"Actuals: {predictions[var_name]['actuals'][:5]}")

def plot_training_losses(losses: Dict[str, list], config) -> None:
    """Plot training losses for each dependent variable"""
    plt.figure(figsize=(12, 6))
    for var_name in config.DEPENDENT_VARIABLES:
        plt.plot(losses[var_name], label=var_name)
    
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 