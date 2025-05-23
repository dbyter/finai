import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def validation_step(self, batch, batch_idx):
    # Forward pass and loss calculation
    output = self(batch)
    loss = output.loss
    self.log('val_loss', loss)
    
    # Calculate additional metrics
    predictions = output.prediction
    actuals = batch['target']
    
    # Calculate MAE and RMSE for each target variable
    for i, var_name in enumerate(self.config.DEPENDENT_VARIABLES):
        mae = torch.mean(torch.abs(predictions[..., i] - actuals[..., i]))
        rmse = torch.sqrt(torch.mean((predictions[..., i] - actuals[..., i])**2))
        
        self.log(f'val_mae_{var_name}', mae)
        self.log(f'val_rmse_{var_name}', rmse)
    
    return loss 