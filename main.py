# Jinni Main Application - Orchestrates the entire AI system
import sys
import logging
import threading
import time
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JinniAISystem:
    def __init__(self):
        self.is_running = False
        self.data_fetcher = None
        self.models = {}
        self.accuracy_tracker = {}
        logger.info("Jinni AI System initialized")
    
    def initialize_components(self):
        try:
            from data_scraper.nse_bse_fetcher import NSEBSEDataFetcher
            from ml_models.lstm_model import LSTMStockPredictor
            
            self.data_fetcher = NSEBSEDataFetcher()
            logger.info("Data fetcher initialized")
            
            self.models['lstm'] = LSTMStockPredictor()
            logger.info("ML models initialized")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def fetch_and_process_data(self):
        while self.is_running:
            try:
                stocks = self.data_fetcher.get_all_nse_stocks()
                logger.info(f"Fetched {len(stocks)} stocks")
                
                for idx, stock in stocks.iterrows():
                    symbol = stock.get('symbol')
                    if symbol:
                        quote = self.data_fetcher.get_live_quote(symbol)
                        if quote:
                            self.process_stock_data(symbol, quote)
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in data fetching loop: {e}")
                time.sleep(5)
    
    def process_stock_data(self, symbol, data):
        try:
            logger.debug(f"Processing {symbol}")
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def start(self):
        logger.info("Starting Jinni AI System...")
        
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return
        
        self.is_running = True
        
        data_thread = threading.Thread(target=self.fetch_and_process_data, daemon=True)
        data_thread.start()
        
        logger.info("Jinni is now running and learning continuously...")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        logger.info("Stopping Jinni AI System...")
        self.is_running = False

if __name__ == "__main__":
    jinni = JinniAISystem()
    jinni.start()
