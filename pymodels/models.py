import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from typing import Dict, List
import logging
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class StockPredictor(nn.Module):
    def __init__(self, config):
        super(StockPredictor, self).__init__()
        self.config = config
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(len(config.FEATURES), 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, len(config.DEPENDENT_VARIABLES))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.bias = nn.Parameter(torch.zeros(len(config.DEPENDENT_VARIABLES)))
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_dependent_variables, d_model, lookback_window = 21):
        super(TransformerModel, self).__init__()
        # Create an embedding layer
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len = lookback_window)
        # Create a transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        # Replace single output projection with two layers
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),                    # smoother; leaves some gradient everywhere
            nn.Linear(d_model, n_dependent_variables)
        )

    def forward(self, src):

        src = self.input_projection(src) # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        # src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, d_model]
        output = self.transformer_encoder(src) # [batch_size, seq_len, d_model]
        seq_repr = output.mean(dim=1) # [batch_size, d_model]
        prediction = self.head(seq_repr) # [batch_size, n_dependent_variables]

        # Dimension debug:
        # logger.info(f"src shape: {src.shape}") # [batch_size, seq_len, n_features]
        # logger.info(f"output shape pre output projection: {output.shape}") # [batch_size, seq_len, d_model]
        # logger.info(f"seq_repr shape: {seq_repr.shape}") # [batch_size, d_model]
        return prediction

class LinearBaseline(nn.Module):
    def __init__(self, n_features=10, seq_len=7, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(seq_len * n_features, n_targets)

    def forward(self, x):           # [B, 7, 10]
        return self.fc(x.flatten(1))

class ModelTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
    
    def train_basic_model(self, model: StockPredictor, train_loader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:
        """Train the basic model"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        losses = {var: [] for var in self.config.DEPENDENT_VARIABLES}
        
        for epoch in range(self.config.NUM_EPOCHS):
            total_losses = {var: 0 for var in self.config.DEPENDENT_VARIABLES}
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # TARGET_COL = 0        # index of the feature where you believe the return lives
                                    # (use the positive index if easier)
                # logger.info(f"batch_X[0]: {batch_X[0, :, :].shape}")
                # logger.info(f"batch_y[0]: {batch_y[0].shape}")
                # compare every element of that feature across the 7 steps with batch_y
                # diff = (batch_X[:, -1, TARGET_COL] - batch_y.unsqueeze(1)).abs().max()
                # print("max |x - y| :", diff.item())
                optimizer.zero_grad()
                # logger.info (f"Batch X shape: {batch_X.shape}")
                # logger.info (f"Batch X[0, :, :] shape: {batch_X[0, :, :].unsqueeze(0).shape}")
                # logger.info(f"Model with data point {batch_X[0, :, :].unsqueeze(0)}, forward: {model(batch_X[0, :, :].unsqueeze(0))}")
                # logger.info(f"Model with data point {batch_X[1, :, :].unsqueeze(0)}, forward: {model(batch_X[1, :, :].unsqueeze(0))}")
                output = model(batch_X)
                # logger.info(f"output shape: {output.shape}")
                # logger.info(f"batch_y shape: {batch_y.shape}")
                # Delta difference between output and batch_y
                loss = 0
                for i, var_name in enumerate(self.config.DEPENDENT_VARIABLES):
                    var_loss = self.criterion(output[:, i], batch_y[:, i])
                    total_losses[var_name] += var_loss.item()
                    loss += var_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item()
                print(f"grad-norm: {total_grad_norm:.5f}")
                optimizer.step()
                batch_count += 1
            
            # Log epoch losses
            for var_name in self.config.DEPENDENT_VARIABLES:
                avg_loss = total_losses[var_name] / batch_count
                losses[var_name].append(avg_loss)
                logger.info(f'Epoch {epoch} - {var_name} Loss: {avg_loss:.6f}')
        
        return losses
    
    def evaluate_basic_model(self, model: StockPredictor, test_loader: torch.utils.data.DataLoader) -> Dict[str, Dict]:
        """Evaluate the basic model and generate HTML report"""
        model.eval()
        metrics = {var: {'predictions': [], 'actuals': [], 'mae': 0, 'rmse': 0} 
                  for var in self.config.DEPENDENT_VARIABLES}
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_X)
                
                for i, var_name in enumerate(self.config.DEPENDENT_VARIABLES):
                    pred = predictions[:, i]
                    actual = batch_y[:, i]
                    
                    metrics[var_name]['predictions'].extend(pred.cpu().numpy())
                    metrics[var_name]['actuals'].extend(actual.cpu().numpy())
        
        # Calculate final metrics
        for var_name in self.config.DEPENDENT_VARIABLES:
            pred = torch.tensor(metrics[var_name]['predictions'])
            actual = torch.tensor(metrics[var_name]['actuals'])
            
            metrics[var_name]['mae'] = torch.abs(pred - actual).mean().item()
            metrics[var_name]['rmse'] = torch.sqrt(torch.mean((pred - actual) ** 2)).item()
        
        # Create HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "model_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <title>Model Performance Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ 
                    background-color: #f5f5f5;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                .plot-container {{ 
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ 
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Model Performance Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        # Add metrics table
        html_content += """
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                </tr>
        """
        
        for var_name in self.config.DEPENDENT_VARIABLES:
            html_content += f"""
                <tr>
                    <td>{var_name}</td>
                    <td>{metrics[var_name]['mae']:.4f}</td>
                    <td>{metrics[var_name]['rmse']:.4f}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Create and save plots for each variable
        for var_name in self.config.DEPENDENT_VARIABLES:
            # Create prediction vs actual plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=metrics[var_name]['actuals'],
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=metrics[var_name]['predictions'],
                name='Predicted',
                line=dict(color='red')
            ))
            fig.update_layout(
                title=f'{var_name} - Predictions vs Actuals',
                xaxis_title='Time Steps',
                yaxis_title='Value',
                showlegend=True
            )
            
            # Create scatter plot
            scatter_fig = px.scatter(
                x=metrics[var_name]['actuals'],
                y=metrics[var_name]['predictions'],
                labels={'x': 'Actual', 'y': 'Predicted'},
                title=f'{var_name} - Actual vs Predicted Scatter Plot'
            )
            scatter_fig.add_trace(go.Scatter(
                x=[min(metrics[var_name]['actuals']), max(metrics[var_name]['actuals'])],
                y=[min(metrics[var_name]['actuals']), max(metrics[var_name]['actuals'])],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
            
            # Add plots to HTML
            html_content += f"""
                <div class="plot-container">
                    <h2>{var_name} Analysis</h2>
                    {fig.to_html(full_html=False)}
                    {scatter_fig.to_html(full_html=False)}
                </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report locally
        report_path = os.path.join(report_dir, f"model_report_{timestamp}.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Model performance report saved locally to: {report_path}")
        
        # Save HTML report to S3
        try:
            s3_client = boto3.client('s3')
            s3_key = f'reports/model_report_{timestamp}.html'
            
            # Upload the HTML content directly to S3
            s3_client.put_object(
                Bucket=self.config.S3_BUCKET,
                Key=s3_key,
                Body=html_content,
                ContentType='text/html'
            )
            
            logger.info(f"Model performance report saved to S3: s3://{self.config.S3_BUCKET}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to save report to S3: {str(e)}")
        
        return metrics

class TFTModel(pl.LightningModule):
    def __init__(self, config, training_dataset):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["training_dataset"])

        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate        = config.TFT_LEARNING_RATE,
            hidden_size          = config.TFT_HIDDEN_SIZE,
            attention_head_size  = config.TFT_ATTENTION_HEADS,
            dropout              = config.TFT_DROPOUT,
            hidden_continuous_size = config.TFT_HIDDEN_CONTINUOUS_SIZE,
            loss                 = QuantileLoss(),
            log_interval         = 10,
            reduce_on_plateau_patience = 4,
        )

    # let Lightning call this with x (and optionally y)
    def forward(self, x):
        return self.model(x)

    # consolidated step for train / val / test
    def _shared_step(self, batch, stage: str):
        if len(batch) == 3:
            x, y, weight = batch
        else:
            x, y        = batch
            weight      = None

        out   = self(x, y)        # ← x is the dict, y is tensor
        loss  = out["loss"]       # already computed because we passed y
        self.log(f"{stage}_loss", loss, prog_bar=True)

        # extra metrics
        preds = out["prediction"]
        mae   = torch.mean(torch.abs(preds - y))
        rmse  = torch.sqrt(torch.mean((preds - y) ** 2))
        self.log(f"{stage}_mae",  mae,  prog_bar=True)
        self.log(f"{stage}_rmse", rmse, prog_bar=True)
        return loss

    def training_step  (self, batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.TFT_LEARNING_RATE)
        return {"optimizer": opt, "monitor": "val_loss"}
