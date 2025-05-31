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
from .config import ModelConfig
logger = logging.getLogger(__name__)


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
        # logger.info(f"x shape: {x.shape}")
        # logger.info(f"pe shape: {self.pe.shape}")
        # logger.info(f"pe slice shape: {self.pe[:, :x.size(1), :].shape}")
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_dependent_variables, d_model, lookback_window=21, 
                 n_years=11, n_months=12):
        super(TransformerModel, self).__init__()
        # Store temporal dimensions
        self.n_years = n_years
        self.n_months = n_months
        self.n_temporal_features = n_years + n_months
        
        # Get the number of base features and tickers from the total feature count
        config = ModelConfig()
        self.n_tickers = len(config.TICKERS)
        # Subtract number of tickers and temporal features
        self.n_base_features = n_features - self.n_tickers - self.n_temporal_features
        
        # Create separate projections for features, tickers, and temporal features
        self.feature_projection = nn.Linear(self.n_base_features, d_model // 2)  # 50% for base features
        self.ticker_projection = nn.Linear(self.n_tickers, d_model // 4)  # 25% for tickers
        
        # Year and month embeddings (sharing the remaining 25%)
        self.year_embedding = nn.Embedding(n_years, d_model // 8)  # Year embedding
        self.month_embedding = nn.Embedding(n_months, d_model // 8)  # Month embedding
        
        # Create a positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback_window)
        
        # Create a transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_dependent_variables)
        )

    def forward(self, src):
        # Split input into base features, year features, month features, and ticker encoding
        base_features = src[:, :, :self.n_base_features]  # Core features excluding time and ticker
        month_features = src[:, :, self.n_base_features:self.n_base_features+self.n_months]  # Month features
        year_features = src[:, :, self.n_base_features+self.n_months:self.n_base_features+self.n_temporal_features]  # Year features
        ticker_features = src[:, :, -self.n_tickers:]  # Ticker features at the end
        
        # Extract year and month indices from one-hot encoded features
        months = torch.argmax(month_features, dim=-1)  # Convert one-hot to indices
        years = torch.argmax(year_features, dim=-1)  # Convert one-hot to indices
        
        # Project features and ticker encoding separately
        base_proj = self.feature_projection(base_features)
        ticker_proj = self.ticker_projection(ticker_features)
        
        # Get temporal embeddings
        year_proj = self.year_embedding(years)   # [batch, seq_len, d_model//8]
        month_proj = self.month_embedding(months)  # [batch, seq_len, d_model//8]
        
        # Combine all projections
        combined = torch.cat([base_proj, ticker_proj, year_proj, month_proj], dim=-1)
        
        # Add positional encoding
        src = self.pos_encoder(combined)
        
        # Pass through transformer
        output = self.transformer_encoder(src)
        
        # Pool sequence dimension using attention
        attn_weights = self.attention_pool(output)
        seq_repr = torch.sum(output * attn_weights, dim=1)
        
        # Generate prediction
        prediction = self.head(seq_repr)
        
        return prediction


class LinearBaseline(nn.Module):
    def __init__(self, n_features=10, seq_len=7, n_targets=4):
        super().__init__()
        self.fc = nn.Linear(seq_len * n_features, n_targets)

    def forward(self, x):           # [B, seq_len, n_targets]
        return self.fc(x.flatten(1))

class ModelTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()

    def gaussian_nll_loss(pred, target, eps=1e-6):
        mean = pred[:, 0]
        log_sigma = pred[:, 1]
        sigma = torch.exp(log_sigma) + eps

        # Negative log likelihood
        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + ((target - mean) ** 2) / (2 * sigma**2)
        return loss.mean()
    def train_basic_model(self, model: TransformerModel, train_loader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:
        """Train the basic model"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        losses = {var: [] for var in self.config.DEPENDENT_VARIABLES}
        
        for epoch in range(self.config.NUM_EPOCHS):
            print (f"Training percentage complete: {epoch / self.config.NUM_EPOCHS * 100:.2f}%, # of epochs: {epoch} out of {self.config.NUM_EPOCHS}")
            total_losses = {var: 0 for var in self.config.DEPENDENT_VARIABLES}
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

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
                    # var_loss = self.gaussian_nll_loss(output[:, i], batch_y[:, i])
                    total_losses[var_name] += var_loss.item()
                    loss += var_loss
                
                # loss = self.gaussian_nll_loss(output, batch_y)
                loss.backward()
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item()
                # logger.info(f"Average grad norm: {total_grad_norm / len(model.parameters())}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                batch_count += 1
            
            # Log epoch losses
            for var_name in self.config.DEPENDENT_VARIABLES:
                avg_loss = total_losses[var_name] / batch_count
                losses[var_name].append(avg_loss)
                logger.info(f'Epoch {epoch} - Loss for {var_name}: {avg_loss:.6f}')
        
        return losses
    
    def evaluate_basic_model(self, model: TransformerModel, test_loader: torch.utils.data.DataLoader, train_loader: torch.utils.data.DataLoader = None, training_losses: Dict[str, List[float]] = None) -> Dict[str, Dict]:
        """Evaluate the basic model and generate HTML report"""
        model.eval()
        metrics = {
            'train': {var: {'predictions': [], 'actuals': [], 'mae': 0, 'rmse': 0} 
                     for var in self.config.DEPENDENT_VARIABLES},
            'test': {var: {'predictions': [], 'actuals': [], 'mae': 0, 'rmse': 0} 
                    for var in self.config.DEPENDENT_VARIABLES}
        }
        
        # Evaluate on training data if loader is provided
        if train_loader is not None:
            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = model(batch_X)
                    logger.info(f"Predictions shape: {predictions.shape}")
                    
                    for i, var_name in enumerate(self.config.DEPENDENT_VARIABLES):
                        pred = predictions[:, i]
                        actual = batch_y[:, i]
                        
                        metrics['train'][var_name]['predictions'].extend(pred.cpu().numpy())
                        metrics['train'][var_name]['actuals'].extend(actual.cpu().numpy())
        
        # Evaluate on test data
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_X)
                
                for i, var_name in enumerate(self.config.DEPENDENT_VARIABLES):
                    pred = predictions[:, i]
                    actual = batch_y[:, i]
                    
                    metrics['test'][var_name]['predictions'].extend(pred.cpu().numpy())
                    metrics['test'][var_name]['actuals'].extend(actual.cpu().numpy())
        
        # Calculate final metrics for both train and test
        for split in ['train', 'test']:
            if split == 'train' and train_loader is None:
                continue
                
            for var_name in self.config.DEPENDENT_VARIABLES:
                logger.info(f"Calculating metrics for {var_name} in {split} split")
                pred = torch.tensor(metrics[split][var_name]['predictions'])
                actual = torch.tensor(metrics[split][var_name]['actuals'])

                # Log percentiles of predicted and actual values
                # logger.info(f"Percentiles for {var_name} in {split} split: {pred.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()}")
                # logger.info(f"Percentiles for {var_name} in {split} split: {actual.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()}")
                
                metrics[split][var_name]['mae'] = torch.abs(pred - actual).mean().item()
                metrics[split][var_name]['rmse'] = torch.sqrt(torch.mean((pred - actual) ** 2)).item()
        
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
                .train-metrics {{ background-color: #e6f3ff; }}
                .test-metrics {{ background-color: #fff0f0; }}
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
                    <th>Split</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                </tr>
        """
        
        for var_name in self.config.DEPENDENT_VARIABLES:
            if train_loader is not None:
                html_content += f"""
                    <tr class="train-metrics">
                        <td>{var_name}</td>
                        <td>Train</td>
                        <td>{metrics['train'][var_name]['mae']:.4f}</td>
                        <td>{metrics['train'][var_name]['rmse']:.4f}</td>
                    </tr>
                """
            html_content += f"""
                <tr class="test-metrics">
                    <td>{var_name}</td>
                    <td>Test</td>
                    <td>{metrics['test'][var_name]['mae']:.4f}</td>
                    <td>{metrics['test'][var_name]['rmse']:.4f}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Add training loss plots if available
        if training_losses is not None:
            html_content += """
                <h2>Training Loss Over Time</h2>
            """
            # Create a single figure for all losses
            loss_fig = go.Figure()
            
            # Add each variable's loss as a separate trace
            for var_name in self.config.DEPENDENT_VARIABLES:
                if var_name in training_losses:
                    loss_fig.add_trace(go.Scatter(
                        y=training_losses[var_name],
                        name=f'{var_name} Loss',
                        mode='lines'
                    ))
            
            loss_fig.update_layout(
                title='Training Loss Over Time for All Variables',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                showlegend=True,
                hovermode='x unified'
            )
            
            html_content += f"""
                <div class="plot-container">
                    {loss_fig.to_html(full_html=False)}
                </div>
            """
        
        # Create and save plots for test data only
        for var_name in self.config.DEPENDENT_VARIABLES:
            # Create prediction vs actual plot for test data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=metrics['test'][var_name]['actuals'],
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=metrics['test'][var_name]['predictions'],
                name='Predicted',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{var_name} - Test Predictions vs Actuals',
                xaxis_title='Time Steps',
                yaxis_title='Value',
                showlegend=True
            )
            
            # Create scatter plot for test data
            scatter_fig = px.scatter(
                x=metrics['test'][var_name]['actuals'],
                y=metrics['test'][var_name]['predictions'],
                labels={'x': 'Actual', 'y': 'Predicted'},
                title=f'{var_name} - Test Actual vs Predicted Scatter Plot'
            )
            
            scatter_fig.add_trace(go.Scatter(
                x=[min(metrics['test'][var_name]['actuals']), max(metrics['test'][var_name]['actuals'])],
                y=[min(metrics['test'][var_name]['actuals']), max(metrics['test'][var_name]['actuals'])],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
            
            # Add plots to HTML
            html_content += f"""
                <div class="plot-container">
                    <h2>{var_name} Test Analysis</h2>
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