import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from local_storage import StorageManager
except ImportError as e:
    print(f"Warning: Could not import storage manager: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Enterprise performance tracking system.
    Tracks prediction accuracy, model performance, and system metrics.
    Stores historical data for audit trail and performance analysis.
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.logger = logger
        self.storage = StorageManager()
        self.logger.info("Performance Tracker initialized")
    
    def track_prediction(self, prediction: Dict, actual_result: Dict) -> Dict:
        """
        Track a prediction and its actual result.
        Calculate accuracy and other metrics.
        """
        try:
            symbol = prediction.get('symbol', '')
            predicted_direction = prediction.get('direction', None)
            predicted_price = prediction.get('price_target', 0)
            confidence = prediction.get('confidence', 0)
            
            actual_direction = actual_result.get('direction', None)
            actual_price = actual_result.get('price', 0)
            
            # Calculate accuracy
            directional_accuracy = 1.0 if predicted_direction == actual_direction else 0.0
            price_error = abs(predicted_price - actual_price) if predicted_price else 0
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'directional_accuracy': directional_accuracy,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'price_error': price_error,
                'confidence': confidence,
                'correct': directional_accuracy > 0.5,
                'confidence_calibration': directional_accuracy - confidence  # Should be ~0 for well-calibrated
            }
            
            # Save to storage
            self.storage.save_performance_record(performance_record)
            self.logger.info(f"Tracked prediction: {symbol}, accuracy={directional_accuracy}")
            
            return performance_record
            
        except Exception as e:
            self.logger.error(f"Error tracking prediction: {e}")
            return {}
    
    def get_prediction_accuracy(self, symbol: str = None, days: int = 30) -> Dict:
        """
        Get prediction accuracy metrics for a symbol or entire system.
        """
        try:
            records = self.storage.get_performance_records(symbol=symbol, days=days)
            
            if not records or len(records) == 0:
                self.logger.info("No performance records found")
                return {}
            
            accuracies = [r.get('directional_accuracy', 0) for r in records]
            confidences = [r.get('confidence', 0) for r in records]
            price_errors = [r.get('price_error', 0) for r in records]
            
            metrics = {
                'total_predictions': len(records),
                'accuracy': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'correct_predictions': sum([r.get('correct', False) for r in records]),
                'win_rate': sum([r.get('correct', False) for r in records]) / len(records),
                'avg_confidence': np.mean(confidences),
                'avg_price_error': np.mean(price_errors),
                'max_price_error': np.max(price_errors),
                'sharpe_ratio': self._calculate_sharpe_ratio(records),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Accuracy: {metrics['accuracy']:.3f}, Win Rate: {metrics['win_rate']:.1%}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, records: List[Dict]) -> float:
        """
        Calculate Sharpe ratio for prediction returns.
        """
        try:
            if not records:
                return 0
            
            returns = []
            for r in records:
                # Returns based on directional accuracy
                accuracy = r.get('directional_accuracy', 0)
                confidence = r.get('confidence', 0)
                
                # Risk-adjusted return
                ret = (accuracy - 0.5) * confidence  # Excess return from 50% baseline
                returns.append(ret)
            
            if len(returns) < 2:
                return 0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_rate = 0.06 / 252  # Daily equivalent
            
            if std_return > 0:
                sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
            else:
                sharpe = 0
            
            return sharpe
            
        except Exception as e:
            self.logger.debug(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def get_model_performance(self, model_name: str = None, days: int = 30) -> Dict:
        """
        Get overall model performance metrics.
        """
        try:
            records = self.storage.get_performance_records(days=days)
            
            if not records:
                return {}
            
            # Group by model if specified
            if model_name:
                records = [r for r in records if r.get('model') == model_name]
            
            # Calculate metrics
            accuracies = [r.get('directional_accuracy', 0) for r in records]
            correctly_predicted = sum([r.get('correct', False) for r in records])
            
            monthly_accuracy = {
                'date': datetime.now().date().isoformat(),
                'total_predictions': len(records),
                'correct_predictions': correctly_predicted,
                'accuracy': np.mean(accuracies) if accuracies else 0,
                'win_rate': correctly_predicted / len(records) if records else 0,
                'std_dev': np.std(accuracies) if len(accuracies) > 1 else 0
            }
            
            # Save performance snapshot
            self.storage.save_performance_snapshot(monthly_accuracy)
            self.logger.info(f"Model performance: {monthly_accuracy}")
            
            return monthly_accuracy
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}
    
    def get_top_performing_stocks(self, limit: int = 10) -> List[Dict]:
        """
        Get stocks with best prediction accuracy.
        """
        try:
            records = self.storage.get_performance_records(days=90)
            
            if not records:
                return []
            
            # Group by symbol
            stock_stats = {}
            for r in records:
                symbol = r.get('symbol', '')
                if symbol not in stock_stats:
                    stock_stats[symbol] = {'count': 0, 'correct': 0, 'total_error': 0}
                
                stock_stats[symbol]['count'] += 1
                if r.get('correct', False):
                    stock_stats[symbol]['correct'] += 1
                stock_stats[symbol]['total_error'] += r.get('price_error', 0)
            
            # Calculate accuracy for each stock
            top_stocks = []
            for symbol, stats in stock_stats.items():
                if stats['count'] >= 5:  # Minimum 5 predictions
                    accuracy = stats['correct'] / stats['count']
                    avg_error = stats['total_error'] / stats['count']
                    
                    top_stocks.append({
                        'symbol': symbol,
                        'predictions': stats['count'],
                        'accuracy': accuracy,
                        'avg_price_error': avg_error
                    })
            
            # Sort by accuracy
            top_stocks.sort(key=lambda x: x['accuracy'], reverse=True)
            
            self.logger.info(f"Top stocks by accuracy: {len(top_stocks)} stocks")
            return top_stocks[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting top stocks: {e}")
            return []
    
    def get_performance_trends(self, days: int = 90) -> Dict:
        """
        Get performance trends over time.
        Shows how accuracy has evolved.
        """
        try:
            records = self.storage.get_performance_records(days=days)
            
            if not records:
                return {}
            
            # Group by day
            daily_stats = {}
            for r in records:
                timestamp = r.get('timestamp', '')
                date = timestamp.split('T')[0] if 'T' in timestamp else timestamp
                
                if date not in daily_stats:
                    daily_stats[date] = {'count': 0, 'correct': 0}
                
                daily_stats[date]['count'] += 1
                if r.get('correct', False):
                    daily_stats[date]['correct'] += 1
            
            # Calculate daily accuracies
            trend = []
            for date in sorted(daily_stats.keys()):
                stats = daily_stats[date]
                accuracy = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
                trend.append({
                    'date': date,
                    'daily_accuracy': accuracy,
                    'predictions': stats['count']
                })
            
            self.logger.info(f"Performance trend: {len(trend)} days")
            return {'trend': trend, 'latest_accuracy': trend[-1]['daily_accuracy'] if trend else 0}
            
        except Exception as e:
            self.logger.error(f"Error getting trends: {e}")
            return {}
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        """
        try:
            accuracy_30d = self.get_prediction_accuracy(days=30)
            accuracy_90d = self.get_prediction_accuracy(days=90)
            model_perf = self.get_model_performance(days=90)
            top_stocks = self.get_top_performing_stocks()
            trends = self.get_performance_trends(days=90)
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'accuracy_30_days': accuracy_30d,
                'accuracy_90_days': accuracy_90d,
                'model_performance': model_perf,
                'top_stocks': top_stocks,
                'trends': trends
            }
            
            # Save report
            self.storage.save_performance_report(report)
            self.logger.info("Performance report generated")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {}


if __name__ == "__main__":
    tracker = PerformanceTracker()
    
    # Example usage
    prediction = {'symbol': 'TCS.NS', 'direction': 'UP', 'price_target': 4500, 'confidence': 0.85}
    actual = {'direction': 'UP', 'price': 4550}
    
    record = tracker.track_prediction(prediction, actual)
    print("Tracked:", record)
    
    accuracy = tracker.get_prediction_accuracy('TCS.NS')
    print("Accuracy:", accuracy)
