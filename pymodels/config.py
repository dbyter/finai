from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    # Data parameters
    TRAIN_START_DATE: str = "2015-01-01"
    TRAIN_END_DATE: str = "2024-07-01"
    TEST_START_DATE: str = "2024-07-02"
    TEST_END_DATE: str = "2025-12-31"
    
    # Feature configuration
    FEATURES = [
        'Return', 
        'Volume', 
        'Close', 
        'Open', 
        'High', 
        'Low', 
        'RSI', 
        'cash', 
        'ebitda', 
        'total_revenue', 
        'gross_profit', 
        'total_assets', 
        'total_liabilities',
        'OBV', 
        '30_day_min', 
        '30_day_max', 
        'accounts_payable',
        'EMA_10',
        'EMA_20',
        'EMA_50',
        'EMA_100',
        'EMA_200',
        'Past_7_Day_Return_Volatility',
        'Past_14_Day_Return_Volatility',
        'Past_28_Day_Return_Volatility',
        'SPY_7day_return',
        'SPY_14day_return',
        'SPY_21day_return',
        'SPY_28day_return',
        'SPY_7day_return_stddev',
        'SPY_14day_return_stddev',
        'SPY_21day_return_stddev',
        'SPY_28day_return_stddev',
        'NASDAQ_7day_return',
        'NASDAQ_14day_return',
        'NASDAQ_21day_return',
        'NASDAQ_28day_return',
        'NASDAQ_7day_return_stddev',
        'NASDAQ_14day_return_stddev',
        'NASDAQ_21day_return_stddev',
        #  'Next_7_Day_Return', 
        #  'Next_7_Day_Return_StdDev',
         #'Next_7_Day_Return', 'Next_14_Day_Return', 'Next_21_Day_Return', 'Next_28_Day_Return', 'Next_7_Day_Return_StdDev', 'Next_14_Day_Return_StdDev', 'Next_21_Day_Return_StdDev', 'Next_28_Day_Return_StdDev'
    ]
    HOT_ENCODING_FEATURES = ['Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 'Is_June', 'Is_July', 'Is_August', 'Is_September', 'Is_October', 'Is_November', 'Is_December']
    HOT_ENCODING_FEATURES += ['Is_2015', 'Is_2016', 'Is_2017', 'Is_2018', 'Is_2019', 'Is_2020', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024', 'Is_2025']

    # DEPENDENT_VARIABLES = ['Next_Day_Return', 'Next_7_Day_Return', 'Next_14_Day_Return', 'Next_21_Day_Return', 'Next_28_Day_Return', 'Next_7_Day_Return_StdDev', 'Next_14_Day_Return_StdDev', 'Next_21_Day_Return_StdDev', 'Next_28_Day_Return_StdDev']
    DEPENDENT_VARIABLES = ['Next_7_Day_Return', 'Next_7_Day_Return_StdDev']
    LOOKBACK_WINDOW: int = 21
    TICKERS =  ['KB', 'VZ', 'MRK', 'ASX', 'RDY', 'TD', 'SAN', 'HUM', 'BTI', 'JPM', 'SHG', 'PRU', 'MS', 'CM', 'EBR', 'BSAC', 'WMT', 'V', 'WFC', 'ERIC', 'BRK-A', 'TTE', 'AMZN', 'STLA', 'WIT', 'GS', 'KSPI', 'GOOG', 'VALE', 'PEP', 'BHP', 'IBN', 'FMX', 'RIO', 'TMUS', 'JD', 'AAPL', 'SMFG', 'TAK', 'HDB', 'BIDU', 'CSCO', 'GM', 'TEF', 'E', 'RY', 'BMO', 'YPF', 'NVO', 'BUD', 'NMR', 'IX', 'PKX', 'CMCSA', 'COP', 'PDD', 'PFE', 'BP', 'NTES', 'ABBV', 'BCH', 'UNH', 'EQNR', 'PG', 'UL', 'ORCL', 'NFLX', 'CHTR', 'CVX', 'MUFG', 'JNJ', 'VIV', 'CHT', 'TLK', 'AVGO', 'EC', 'BABA', 'KOF', 'XOM', 'TM', 'HD', 'TSM', 'BAC', 'AMX', 'HSBC', 'NVS', 'SHEL', 'HMC', 'MFG', 'ITUB', 'BSBR', 'SONY', 'VOD', 'C', 'NVDA', 'UMC', 'T', 'INTC', 'ABEV', 'SUZ', 'PBR', 'BBD', 'META', 'DIS', 'MSFT']
    # TICKERS = ['AAPL']
    FEATURE_COUNT = len(FEATURES) + len(HOT_ENCODING_FEATURES) + len(TICKERS) 
    
    # Model parameters
    BATCH_SIZE: int = 50
    NUM_EPOCHS: int = 400
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.2
    
    # TFT specific parameters
    TFT_HIDDEN_SIZE: int = 16
    TFT_ATTENTION_HEADS: int = 4
    TFT_DROPOUT: float = 0.1
    TFT_HIDDEN_CONTINUOUS_SIZE: int = 8
    TFT_LEARNING_RATE: float = 0.03
    
    # AWS configuration
    S3_BUCKET: str = 'tradingmodelsahmed'
    S3_MODEL_KEY: str = 'models/model_with_metrics.pth' 