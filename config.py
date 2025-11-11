# Jinni Configuration File
import os
from datetime import timedelta

class Config:
    # Application Settings
    APP_NAME = "Jinni Indian Stock Market AI"
    VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    
    # Data Fetching
    NSE_BASE_URL = "https://www.nseindia.com"
    BSE_BASE_URL = "https://www.bseindia.com"
    DATA_FETCH_INTERVAL = 60
    
    # Machine Learning
    ML_MODEL_PATH = "models/"
    LSTM_UNITS = 128
    LSTM_SEQUENCE_LENGTH = 60
    GRU_UNITS = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Online Learning
    ONLINE_LEARNING_ENABLED = True
    INCREMENTAL_UPDATE_INTERVAL = 1
    ACCURACY_THRESHOLD = 0.85
    
    # Database
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
    DB_PATH = os.getenv('DB_PATH', 'jinni_data.db')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', 5432)
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'jinni')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'jinni_db')
    
    # Dashboard
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8501))
    DASHBOARD_UPDATE_INTERVAL = 5
    
    # Analytics
    TECHNICAL_INDICATORS = ['SMA', 'EMA', 'RSI', 'MACD', 'BB']
    RISK_ANALYSIS_ENABLED = True
    PORTFOLIO_OPTIMIZATION_ENABLED = True
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'jinni.log'
    
    # Performance
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    CACHE_ENABLED = True
    CACHE_TTL = 300
    
    # API Keys (if needed)
    NSE_API_KEY = os.getenv('NSE_API_KEY', '')
    BSE_API_KEY = os.getenv('BSE_API_KEY', '')
    
    @classmethod
    def get_db_url(cls):
        if cls.DB_TYPE == 'postgres':
            return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        return f"sqlite:///{cls.DB_PATH}"

config = Config()
