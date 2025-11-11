# NSE/BSE Data Fetcher - Custom built
import requests
import pandas as pd
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSEBSEDataFetcher:
    def __init__(self):
        self.nse_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
    
    def get_all_nse_stocks(self):
        try:
            url = f"{self.nse_url}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data.get('data', []))
        except Exception as e:
            logger.error(f"Error: {e}")
        return pd.DataFrame()
    
    def get_live_quote(self, symbol):
        try:
            url = f"{self.nse_url}/api/quote-equity?symbol={symbol}"
            response = self.session.get(url)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
        return None
