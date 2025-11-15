"""
Automated Retraining & Continuous Learning - Autonomous Self-Improvement System

This module enables JINNI to autonomously:
- Select stocks for prediction
- Evaluate prediction accuracy against actual market behavior
- Learn from successes and failures
- Automatically retrain models when accuracy drops
- Continuously improve without manual intervention

Key Features:
- Scheduled prediction validation
- Automatic accuracy tracking
- Intelligent retraining triggers
- A/B testing for model improvements
- Performance monitoring and alerting

Author: JINNI Development Team
Designed for autonomous improvement like BlackRock's Aladdin
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import time
import schedule
import yfinance as yf

# Our custom modules
try:
    from firebase_config import FirebaseManager
    from advanced_models import AdvancedEnsembleModel
    from recommendation_engine import RecommendationEngine
except ImportError:
    print("Warning: Some modules not found. Make sure firebase_config.py, advanced_models.py, and recommendation_engine.py are available.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedRetrainingSystem:
    """
    Autonomous learning system that continuously improves model accuracy.
    
    Features:
    - Automatic stock selection for predictions
    - Real-time accuracy tracking
    - Smart retraining triggers
    - A/B testing for model variants
    - Performance analytics
    """
    
    def __init__(self, firebase_manager: FirebaseManager, 
                 retrain_threshold: float = 0.60,
                 check_interval_hours: int = 24):
        """
        Initialize automated retraining system.
        
        Args:
            firebase_manager: Firebase manager for data storage
            retrain_threshold: Accuracy below this triggers retraining (default 60%)
            check_interval_hours: Hours between accuracy checks (default 24)
        """
        self.firebase = firebase_manager
        self.retrain_threshold = retrain_threshold
        self.check_interval = check_interval_hours
        self.model = None
        self.performance_history = []
        
        logger.info(f"Initialized AutomatedRetrainingSystem with {retrain_threshold:.0%} threshold")
    
    def select_stocks_for_validation(self, n_stocks: int = 50) -> List[str]:
        """
        Autonomously select stocks for prediction validation.
        
        Args:
            n_stocks: Number of stocks to validate
        
        Returns:
            List of stock symbols
        """
        logger.info(f"Selecting {n_stocks} stocks for validation...")
        
        # Get stocks from recent scans
        recent_scans = self.firebase.get_recent_scans(days=7)
        
        if recent_scans:
            # Select diverse stocks from recent scans
            all_symbols = set()
            for scan in recent_scans:
                if 'symbols' in scan:
                    all_symbols.update(scan['symbols'][:10])
            
            selected = list(all_symbols)[:n_stocks]
        else:
            # Default to popular NSE stocks
            selected = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
                'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
                'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'WIPRO.NS'
            ][:n_stocks]
        
        logger.info(f"Selected {len(selected)} stocks: {selected[:5]}...")
        return selected
    
    def validate_predictions(self, symbols: List[str], 
                           prediction_horizon_days: int = 7) -> Dict:
        """
        Validate predictions against actual market performance.
        
        Args:
            symbols: List of stock symbols to validate
            prediction_horizon_days: Days ahead predictions were made for
        
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating predictions for {len(symbols)} stocks...")
        
        # Get predictions made N days ago
        cutoff_date = datetime.now() - timedelta(days=prediction_horizon_days)
        old_predictions = self.firebase.get_predictions_after_date(cutoff_date.isoformat())
        
        if not old_predictions:
            logger.warning("No predictions found for validation")
            return {'accuracy': 0, 'total': 0}
        
        correct_predictions = 0
        total_predictions = 0
        prediction_errors = []
        
        for pred in old_predictions:
            symbol = pred.get('symbol')
            predicted_price = pred.get('predicted_price')
            prediction_date = pred.get('timestamp')
            
            if not all([symbol, predicted_price, prediction_date]):
                continue
            
            # Get actual price
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1mo')
                
                if len(hist) > 0:
                    actual_price = hist['Close'].iloc[-1]
                    original_price = pred.get('current_price', actual_price)
                    
                    # Calculate if direction was correct
                    predicted_direction = 1 if predicted_price > original_price else -1
                    actual_direction = 1 if actual_price > original_price else -1
                    
                    if predicted_direction == actual_direction:
                        correct_predictions += 1
                    
                    # Calculate percentage error
                    error = abs(predicted_price - actual_price) / actual_price
                    prediction_errors.append(error)
                    
                    total_predictions += 1
                    
            except Exception as e:
                logger.debug(f"Error validating {symbol}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_error = np.mean(prediction_errors) if prediction_errors else 1.0
        
        results = {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions,
            'avg_error': avg_error,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Validation complete: {accuracy:.1%} accuracy ({correct_predictions}/{total_predictions})")
        
        # Store validation results
        self.firebase.log_metric('validation_accuracy', accuracy)
        self.performance_history.append(results)
        
        return results
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on performance.
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        if not self.performance_history:
            return False, "No performance history available"
        
        recent_performance = self.performance_history[-1]
        accuracy = recent_performance.get('accuracy', 0)
        
        # Trigger 1: Accuracy below threshold
        if accuracy < self.retrain_threshold:
            return True, f"Accuracy {accuracy:.1%} below threshold {self.retrain_threshold:.1%}"
        
        # Trigger 2: Declining trend (last 3 validations)
        if len(self.performance_history) >= 3:
            recent_accuracies = [p['accuracy'] for p in self.performance_history[-3:]]
            if all(recent_accuracies[i] > recent_accuracies[i+1] for i in range(len(recent_accuracies)-1)):
                return True, f"Declining accuracy trend detected: {recent_accuracies}"
        
        # Trigger 3: High error rate
        avg_error = recent_performance.get('avg_error', 0)
        if avg_error > 0.15:  # 15% average error
            return True, f"High prediction error: {avg_error:.1%}"
        
        return False, "Performance is satisfactory"
    
    def retrain_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Retrain the model with fresh data.
        
        Args:
            training_data: New training data
        
        Returns:
            Dictionary with retraining results
        """
        logger.info("Starting model retraining...")
        start_time = datetime.now()
        
        try:
            # Create new model instance
            new_model = AdvancedEnsembleModel(task='regression', optimize=False)
            
            # Prepare features and target
            if 'target' in training_data.columns:
                X = training_data.drop('target', axis=1)
                y = training_data['target']
            else:
                # Use return as target if not specified
                X = training_data.iloc[:, :-1]
                y = training_data.iloc[:, -1]
            
            # Train the model
            metrics = new_model.train(X, y, optimize_each=False)
            
            # Save the new model
            new_model.save_models('models/retrained/')
            
            # Update current model
            self.model = new_model
            
            # Log retraining
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'metrics': metrics,
                'training_time': training_time,
                'samples': len(training_data),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Retraining complete in {training_time:.1f}s. R²: {metrics['ensemble']['r2']:.3f}")
            
            # Store in Firebase
            self.firebase.log_metric('retrain_r2', metrics['ensemble']['r2'])
            self.firebase.log_metric('retrain_timestamp', result['timestamp'])
            
            return result
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def collect_training_data(self, n_days: int = 90) -> pd.DataFrame:
        """
        Collect fresh training data from market.
        
        Args:
            n_days: Days of historical data to collect
        
        Returns:
            DataFrame with training data
        """
        logger.info(f"Collecting {n_days} days of training data...")
        
        # Select diverse stocks
        symbols = self.select_stocks_for_validation(n_stocks=30)
        
        all_data = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=f'{n_days}d')
                
                if len(hist) > 30:  # Minimum data requirement
                    # Calculate features
                    hist['returns'] = hist['Close'].pct_change()
                    hist['volatility'] = hist['returns'].rolling(window=20).std()
                    hist['ma_20'] = hist['Close'].rolling(window=20).mean()
                    hist['ma_50'] = hist['Close'].rolling(window=50).mean()
                    hist['rsi'] = self._calculate_rsi(hist['Close'], periods=14)
                    
                    # Target: next day return
                    hist['target'] = hist['returns'].shift(-1)
                    
                    # Drop NaN
                    hist = hist.dropna()
                    
                    if len(hist) > 0:
                        features = hist[['returns', 'volatility', 'rsi', 'target']]
                        all_data.append(features)
                        
            except Exception as e:
                logger.debug(f"Error collecting data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected {len(combined_data)} training samples")
            return combined_data
        else:
            logger.warning("No training data collected")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_validation_cycle(self) -> Dict:
        """
        Run complete validation and retraining cycle.
        
        Returns:
            Dictionary with cycle results
        """
        logger.info("="*50)
        logger.info("Starting automated validation cycle")
        logger.info("="*50)
        
        # Step 1: Select stocks
        symbols = self.select_stocks_for_validation()
        
        # Step 2: Validate predictions
        validation_results = self.validate_predictions(symbols)
        
        # Step 3: Check if retraining needed
        should_retrain, reason = self.should_retrain()
        
        results = {
            'validation': validation_results,
            'retrain_triggered': should_retrain,
            'retrain_reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 4: Retrain if needed
        if should_retrain:
            logger.info(f"🔄 Retraining triggered: {reason}")
            
            # Collect fresh data
            training_data = self.collect_training_data()
            
            if len(training_data) > 100:
                retrain_results = self.retrain_model(training_data)
                results['retrain'] = retrain_results
                
                if retrain_results['success']:
                    logger.info("✅ Retraining successful!")
                else:
                    logger.error("❌ Retraining failed!")
            else:
                logger.warning("⚠️  Insufficient training data")
                results['retrain'] = {'success': False, 'error': 'Insufficient data'}
        else:
            logger.info(f"✓ No retraining needed: {reason}")
        
        logger.info("="*50)
        return results
    
    def start_automated_monitoring(self):
        """
        Start continuous monitoring and automated retraining.
        
        This runs indefinitely and checks performance every N hours.
        """
        logger.info(f"🚀 Starting automated monitoring (checking every {self.check_interval}h)")
        
        # Schedule validation cycles
        schedule.every(self.check_interval).hours.do(self.run_validation_cycle)
        
        # Run initial cycle
        self.run_validation_cycle()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get summary of recent performance.
        
        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        return df


# Example usage
if __name__ == "__main__":
    print("Automated Retraining System - Autonomous Learning for JINNI")
    print("="*70)
    
    # Note: This is example usage. In production, initialize with real FirebaseManager
    print("\n⚡ System Capabilities:")
    print("  1. Autonomous stock selection for validation")
    print("  2. Automatic prediction accuracy tracking")
    print("  3. Intelligent retraining triggers:")
    print("     - Accuracy drops below 60%")
    print("     - Declining trend (3+ validations)")
    print("     - High prediction error (>15%)")
    print("  4. Fresh data collection from market")
    print("  5. Automatic model retraining")
    print("  6. Continuous monitoring (24/7)")
    print("\n🎯 This enables JINNI to:")
    print("  ✓ Self-select stocks for evaluation")
    print("  ✓ Learn from market behavior")
    print("  ✓ Improve accuracy autonomously")
    print("  ✓ Never require manual intervention")
    print("\n" + "="*70)
    print("System ready for autonomous deployment!")
