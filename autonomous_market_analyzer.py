import os
import sys
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_models import AdvancedEnsembleModel
    from recommendation_engine import RecommendationEngine
    from risk_analytics import RiskAnalytics
    from automated_retraining import AutomatedRetrainingSystem
    from local_storage import StorageManager
except ImportError as e:
    print(f"Warning: Could not import enterprise modules: {e}")
    print("System will attempt to load modules dynamically")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutonomousMarketAnalyzer:
    """
    Enterprise-level autonomous market analyzer for Indian stock market.
    Continuously scans entire stock universe without manual intervention.
    Integrates AdvancedEnsembleModel, RecommendationEngine, and RiskAnalytics.
    """
    
    def __init__(self, config_path='config.json'):
        """Initialize analyzer with enterprise modules"""
        self.logger = logger
        self.logger.info("Initializing Autonomous Market Analyzer...")
        
        # Initialize storage
        self.storage = StorageManager()
        
        # Load or initialize enterprise modules
        try:
            self.ensemble_model = AdvancedEnsembleModel()
            self.recommendation_engine = RecommendationEngine()
            self.risk_analytics = RiskAnalytics()
            self.retraining_system = AutomatedRetrainingSystem()
            self.logger.info("Enterprise modules loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load enterprise modules: {e}")
            raise
        
        # Configuration
        self.scan_interval = 3600  # Scan every hour (can be adjusted)
        self.analysis_interval = 1800  # Deep analysis every 30 minutes
        self.retraining_interval = 86400  # Retrain models daily
        
        # NSE stock universe - approximately 2000 stocks
        self.nse_stocks = self._load_nse_universe()
        self.logger.info(f"Loaded {len(self.nse_stocks)} stocks from NSE universe")
        
        # Analysis tracking
        self.last_scan = None
        self.last_deep_analysis = None
        self.last_retraining = None
        self.running = False
        self.stop_event = threading.Event()
        
        self.logger.info("Autonomous Market Analyzer initialized successfully")
    
    def _load_nse_universe(self):
        """
        Load NSE stock universe.
        Returns list of NSE stock symbols.
        """
        try:
            # Load from storage if available
            nse_list = self.storage.get_nse_universe()
            if nse_list and len(nse_list) > 100:
                return nse_list
        except:
            pass
        
        # Default NSE universe - major stocks
        nse_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'SBIN.NS',
            'ICICIBANK.NS', 'HDFC.NS', 'HDFC.NS', 'MARUTI.NS', 'BAJAJFINSV.NS',
            'LT.NS', 'ASIANPAINT.NS', 'WIPRO.NS', 'AXISBANK.NS', 'DMART.NS',
            'SUNPHARMA.NS', 'BHARATIARTL.NS', 'JSWSTEEL.NS', 'POWERGRID.NS',
            'HCLTECH.NS', 'DIVISLAB.NS', 'TECHM.NS', 'ULTRACEMCO.NS', 'TATASTEEL.NS',
            'BAJAJHLDNG.NS', 'NESTLEIND.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
            'BAJAJFINSV.NS', 'GAIL.NS', 'NTPC.NS', 'COAL.NS', 'HINDALCO.NS'
        ]
        
        self.storage.save_nse_universe(nse_stocks)
        return nse_stocks
    
    def continuous_scan(self):
        """
        Continuously scan market at regular intervals.
        This runs in a background thread and never stops.
        """
        self.logger.info("Starting continuous market scan...")
        self.running = True
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Regular scan
                if not self.last_scan or (current_time - self.last_scan).seconds >= self.scan_interval:
                    self.logger.info(f"Starting regular scan at {current_time}")
                    self._perform_quick_scan()
                    self.last_scan = current_time
                
                # Deep analysis
                if not self.last_deep_analysis or (current_time - self.last_deep_analysis).seconds >= self.analysis_interval:
                    self.logger.info(f"Starting deep analysis at {current_time}")
                    self._perform_deep_analysis()
                    self.last_deep_analysis = current_time
                
                # Model retraining
                if not self.last_retraining or (current_time - self.last_retraining).seconds >= self.retraining_interval:
                    self.logger.info(f"Starting model retraining at {current_time}")
                    self._retrain_models()
                    self.last_retraining = current_time
                
                # Sleep before next check
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in continuous scan: {e}")
                time.sleep(600)  # Wait 10 minutes before retrying
    
    def _perform_quick_scan(self):
        """
        Perform quick scan of all stocks.
        Generates predictions for each stock.
        """
        try:
            scan_results = []
            batch_size = 50
            
            for i in range(0, len(self.nse_stocks), batch_size):
                batch = self.nse_stocks[i:i+batch_size]
                self.logger.info(f"Scanning batch {i//batch_size + 1}: {len(batch)} stocks")
                
                for symbol in batch:
                    try:
                        # Fetch recent data
                        data = yf.download(symbol, period='1y', progress=False)
                        if len(data) < 50:
                            continue
                        
                        # Generate ensemble prediction
                        prediction = self.ensemble_model.predict(data, symbol)
                        
                        # Get risk metrics
                        risk_metrics = self.risk_analytics.calculate_metrics(data)
                        
                        # Store prediction
                        scan_results.append({
                            'symbol': symbol,
                            'prediction': prediction,
                            'confidence': prediction.get('confidence', 0),
                            'risk_metrics': risk_metrics,
                            'timestamp': datetime.now(),
                            'price': data['Close'].iloc[-1]
                        })
                        
                        # Save to storage
                        self.storage.save_prediction({
                            'symbol': symbol,
                            'prediction': prediction,
                            'risk_metrics': risk_metrics,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        self.logger.debug(f"Error processing {symbol}: {e}")
                        continue
            
            self.logger.info(f"Quick scan completed: {len(scan_results)} stocks analyzed")
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Error in quick scan: {e}")
            return []
    
    def _perform_deep_analysis(self):
        """
        Perform deep technical and fundamental analysis.
        Generates detailed recommendations.
        """
        try:
            recommendations = []
            
            # Get recent predictions from storage
            recent_predictions = self.storage.get_recent_predictions(limit=100)
            
            if not recent_predictions:
                # If no stored predictions, run a scan first
                self._perform_quick_scan()
                recent_predictions = self.storage.get_recent_predictions(limit=100)
            
            # Generate recommendations using recommendation engine
            for pred in recent_predictions:
                try:
                    recommendation = self.recommendation_engine.generate_recommendation(
                        symbol=pred['symbol'],
                        prediction=pred['prediction'],
                        risk_metrics=pred['risk_metrics']
                    )
                    
                    if recommendation and recommendation.get('confidence', 0) > 0.5:
                        recommendations.append(recommendation)
                        
                        # Save recommendation
                        self.storage.save_recommendation({
                            'symbol': pred['symbol'],
                            'recommendation': recommendation,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    self.logger.debug(f"Error generating recommendation for {pred.get('symbol')}: {e}")
                    continue
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            top_recommendations = recommendations[:20]  # Top 20 recommendations (~10% of universe)
            
            self.logger.info(f"Deep analysis completed: {len(top_recommendations)} recommendations generated")
            return top_recommendations
            
        except Exception as e:
            self.logger.error(f"Error in deep analysis: {e}")
            return []
    
    def _retrain_models(self):
        """
        Retrain models with latest market data.
        Called daily for continuous learning.
        """
        try:
            self.logger.info("Starting model retraining...")
            
            # Collect recent data for retraining
            retraining_data = []
            for symbol in self.nse_stocks[:100]:  # Sample for retraining
                try:
                    data = yf.download(symbol, period='1y', progress=False)
                    if len(data) > 50:
                        retraining_data.append({
                            'symbol': symbol,
                            'data': data
                        })
                except:
                    continue
            
            # Retrain ensemble model
            if retraining_data:
                self.ensemble_model.retrain(retraining_data)
                self.logger.info("Ensemble model retrained successfully")
            
            # Retrain automated retraining system
            self.retraining_system.run_retraining(retraining_data)
            self.logger.info("Automated retraining system executed")
            
            # Save retraining metrics
            self.storage.save_retraining_event({
                'timestamp': datetime.now().isoformat(),
                'stocks_trained': len(retraining_data),
                'status': 'success'
            })
            
        except Exception as e:
            self.logger.error(f"Error in model retraining: {e}")
    
    def get_latest_recommendations(self, limit=20):
        """
        Get latest recommendations from storage.
        """
        try:
            return self.storage.get_recommendations(limit=limit)
        except Exception as e:
            self.logger.error(f"Error retrieving recommendations: {e}")
            return []
    
    def get_scan_status(self):
        """
        Get current analysis status.
        """
        return {
            'running': self.running,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'last_deep_analysis': self.last_deep_analysis.isoformat() if self.last_deep_analysis else None,
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'stocks_in_universe': len(self.nse_stocks)
        }
    
    def start(self):
        """
        Start autonomous analyzer in background thread.
        """
        if not self.running:
            self.stop_event.clear()
            analyzer_thread = threading.Thread(target=self.continuous_scan, daemon=True)
            analyzer_thread.start()
            self.logger.info("Autonomous analyzer started")
    
    def stop(self):
        """
        Stop autonomous analyzer gracefully.
        """
        self.stop_event.set()
        self.running = False
        self.logger.info("Autonomous analyzer stopped")


if __name__ == "__main__":
    # Initialize and start analyzer
    analyzer = AutonomousMarketAnalyzer()
    analyzer.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        analyzer.stop()
        print("\nAnalyzer shut down")
