from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    # Data parameters
    TRAIN_START_DATE: str = "2015-01-01"
    TRAIN_END_DATE: str = "2024-12-31"
    TEST_START_DATE: str = "2025-01-01"
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
        'OBV', 
        '30_day_min', 
        '30_day_max', 
        'accounts_payable',
        'EMA_10',
        'EMA_20',
        'EMA_50',
        'EMA_100',
        'EMA_200'
    ]
    HOT_ENCODING_FEATURES = ['Is_2020', 'Is_2021', 'Is_2022', 'Is_2023', 'Is_2024', 'Is_2025', 'Is_January', 'Is_February', 'Is_March', 'Is_April', 'Is_May', 'Is_June', 'Is_July', 'Is_August', 'Is_September', 'Is_October', 'Is_November', 'Is_December']

    DEPENDENT_VARIABLES = ['Next_Day_Return', 'Next_7_Day_Return', 'Next_14_Day_Return', 'Next_21_Day_Return', 'Next_28_Day_Return']
    LOOKBACK_WINDOW: int = 21
    TICKERS =  ['ACI', 'AMGN', 'GOLD', 'HUM', 'BBVA', 'ZTS', 'SBUX', 'SUZ', 'USB', 'ICE', 'NVDA', 'MT', 'AIG', 'LOW', 'SCCO', 'VST', 'STT', 'MMM', 'ITUB', 'BAK', 'CVX', 'DAL', 'ADI', 'BABA', 'ABBV', 'PDD', 'BSAC', 'AVGO', 'CME', 'CRM', 'MPLX', 'PEG', 'ORCL', 'PCG', 'TJX', 'PPL', 'ELV', 'NSC', 'CRBG', 'V', 'NEE', 'MSFT', 'AFL', 'UBS', 'BNS', 'LYG', 'AEP', 'HES', 'RTX', 'ATUS', 'NTES', 'AEE', 'SHG', 'EPD', 'HPQ', 'BIIB', 'ADP', 'SYF', 'BAP', 'HCA', 'TAK', 'NGG', 'ABT', 'WBA', 'PKX', 'FCX', 'BAC', 'CM', 'NVO', 'OKE', 'EXE', 'SONY', 'CTVA', 'APD', 'ET', 'CARR', 'LI', 'STM', 'PARA', 'WMB', 'BDX', 'PSA', 'E', 'GSK', 'DIS', 'OXY', 'AXP', 'GOOG', 'ENB', 'KSPI', 'TU', 'KKR', 'CNQ', 'PYPL', 'TD', 'PGR', 'TME', 'TSM', 'COP', 'URI', 'NUE', 'BHP', 'GS', 'MCD', 'UNH', 'COST', 'SYK', 'GFI', 'UL', 'ACN', 'SLF', 'BRK-A', 'JPM', 'SNY', 'FDX', 'TGT', 'RIO', 'MFC', 'SHW', 'SBS', 'VOD', 'MUFG', 'BCS', 'XOM', 'VTRS', 'VLO', 'SMFG', 'GM', 'DUK', 'EMR', 'LIN', 'PEP', 'SAN', 'REGN', 'DVN', 'DHR', 'WBD', 'BBD', 'KO', 'AMT', 'AMP', 'ALL', 'EXC', 'LEN', 'C', 'MET', 'INTU', 'VG', 'YPF', 'NKE', 'KMI', 'MS', 'JD', 'TXT', 'ED', 'HAL', 'LNG', 'ES', 'FE', 'GEHC', 'GD', 'LYB', 'DOW', 'KB', 'AAPL', 'LVS', 'NVS', 'MDT', 'CP', 'MDLZ', 'BMY', 'ETR', 'ECL', 'MMC', 'FANG', 'CAT', 'ABEV', 'DEO', 'INTC', 'CSCO', 'BP', 'XEL', 'FMS', 'HTHT', 'TCOM', 'ITW', 'BPYPP', 'UNM', 'AER', 'BCH', 'HPE', 'MA', 'EOG', 'CL', 'VZ', 'NEM', 'BK', 'HSBC', 'DB', 'PH', 'GE', 'CRH', 'MO', 'SRE', 'TRP', 'MCK', 'TMO', 'HDB', 'FOXA', 'FTS', 'LRCX', 'NWG', 'SHEL', 'KHC', 'NOC', 'CEG', 'PSX', 'LUV', 'BTI', 'PG', 'VIV', 'WMT', 'CNH', 'BUD', 'EQNR', 'AZO', 'RY', 'CVE', 'AMZN', 'PRU', 'SAP', 'UPS', 'HMC', 'DHI', 'CB', 'THC', 'FIS', 'MTB', 'NXPI', 'ASML', 'PCAR', 'QCOM', 'KMB', 'GIS', 'DTE', 'HD', 'UAL', 'IX', 'META', 'MPC', 'PBR', 'TSLA', 'WM', 'VALE', 'WFC', 'RGA', 'ORLY', 'ADBE', 'RELX', 'BIDU', 'EC', 'MAR', 'KDP', 'BMO', 'WEC', 'WDS', 'GILD', 'AMAT', 'TRV', 'BX', 'IBM', 'DD', 'RSG', 'ZTO', 'DE', 'NTR', 'BIP', 'TSN', 'CNI', 'COF', 'RCI', 'LMT', 'AZN', 'FITB', 'CHT', 'CQP', 'TLK', 'CCEP', 'KR', 'TMUS', 'KLAC', 'UNP', 'NMR', 'STLA', 'JNJ', 'TXN', 'WIT', 'DFS', 'MRK', 'IBN', 'AON', 'BNTX', 'HON', 'PNC', 'TFC', 'IP', 'ADM', 'GMAB', 'ING', 'BLK', 'SPG', 'CSX', 'SO', 'KOF', 'CTSH', 'INFY', 'IMO', 'TM', 'DELL', 'ERIC', 'TEL', 'PLD', 'D', 'SCHW', 'TEF', 'BSBR', 'BCE', 'ETN', 'FUTU', 'APA', 'FMX', 'PFE', 'SLB', 'SPGI', 'MFG', 'CVS', 'TTE', 'LLY', 'CCI', 'MU', 'PM', 'EIX', 'NFLX', 'CMCSA', 'EBR', 'TECK', 'HIG', 'SYY', 'AES', 'UMC', 'RDY', 'ASX', 'AMX', 'SU', 'CI', 'F', 'CMI', 'JCI', 'T', 'CNC', 'BKNG', 'FI', 'CHTR', 'DG']

    FEATURE_COUNT = len(FEATURES) + len(TICKERS) + len(HOT_ENCODING_FEATURES)
    
    # Model parameters
    BATCH_SIZE: int = 1024
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 0.0003
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