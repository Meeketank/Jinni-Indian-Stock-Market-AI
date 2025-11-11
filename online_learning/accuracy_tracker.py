import numpy as np
import time
from datetime import datetime
from typing import List, Dict
import json

class AccuracyTracker:
    """Track and display model accuracy metrics continuously"""
    
    def __init__(self):
        self.accuracy_history = []
        self.prediction_errors = []
        self.timestamps = []
        self.stock_accuracies = {}
        self.overall_accuracy = 0.0
        self.start_time = time.time()
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def update_accuracy(self, predicted_value: float, actual_value: float, stock_symbol: str):
        """Update accuracy with new prediction"""
        error = abs(predicted_value - actual_value)
        percentage_error = (error / actual_value) * 100 if actual_value != 0 else 0
        
        # Consider prediction correct if within 2% of actual
        is_correct = percentage_error < 2.0
        
        self.total_predictions += 1
        if is_correct:
            self.correct_predictions += 1
        
        # Update overall accuracy
        self.overall_accuracy = (self.correct_predictions / self.total_predictions) * 100
        
        # Store history
        self.prediction_errors.append(percentage_error)
        self.accuracy_history.append(self.overall_accuracy)
        self.timestamps.append(datetime.now())
        
        # Update per-stock accuracy
        if stock_symbol not in self.stock_accuracies:
            self.stock_accuracies[stock_symbol] = {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'avg_error': 0.0,
                'errors': []
            }
        
        self.stock_accuracies[stock_symbol]['total'] += 1
        if is_correct:
            self.stock_accuracies[stock_symbol]['correct'] += 1
        
        self.stock_accuracies[stock_symbol]['errors'].append(percentage_error)
        self.stock_accuracies[stock_symbol]['accuracy'] = (
            self.stock_accuracies[stock_symbol]['correct'] / 
            self.stock_accuracies[stock_symbol]['total']
        ) * 100
        self.stock_accuracies[stock_symbol]['avg_error'] = np.mean(
            self.stock_accuracies[stock_symbol]['errors']
        )
    
    def get_current_accuracy(self) -> float:
        """Get current overall accuracy"""
        return self.overall_accuracy
    
    def get_accuracy_improvement_rate(self) -> float:
        """Calculate rate of accuracy improvement per second"""
        if len(self.accuracy_history) < 2:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        
        initial_accuracy = self.accuracy_history[0] if self.accuracy_history[0] > 0 else 0.1
        current_accuracy = self.accuracy_history[-1]
        
        improvement = current_accuracy - initial_accuracy
        rate = improvement / elapsed_time
        
        return rate
    
    def get_statistics(self) -> Dict:
        """Get comprehensive accuracy statistics"""
        elapsed_time = time.time() - self.start_time
        
        stats = {
            'overall_accuracy': self.overall_accuracy,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'average_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'median_error': np.median(self.prediction_errors) if self.prediction_errors else 0.0,
            'min_error': np.min(self.prediction_errors) if self.prediction_errors else 0.0,
            'max_error': np.max(self.prediction_errors) if self.prediction_errors else 0.0,
            'improvement_rate_per_second': self.get_accuracy_improvement_rate(),
            'elapsed_time_seconds': elapsed_time,
            'predictions_per_second': self.total_predictions / elapsed_time if elapsed_time > 0 else 0
        }
        
        return stats
    
    def get_stock_accuracy(self, stock_symbol: str) -> Dict:
        """Get accuracy for specific stock"""
        if stock_symbol in self.stock_accuracies:
            return self.stock_accuracies[stock_symbol]
        return None
    
    def get_all_stock_accuracies(self) -> Dict:
        """Get accuracies for all stocks"""
        return self.stock_accuracies
    
    def get_top_performing_stocks(self, top_n: int = 10) -> List:
        """Get top N stocks by accuracy"""
        sorted_stocks = sorted(
            self.stock_accuracies.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        return sorted_stocks[:top_n]
    
    def get_worst_performing_stocks(self, bottom_n: int = 10) -> List:
        """Get bottom N stocks by accuracy"""
        sorted_stocks = sorted(
            self.stock_accuracies.items(),
            key=lambda x: x[1]['accuracy']
        )
        return sorted_stocks[:bottom_n]
    
    def display_live_stats(self):
        """Display live statistics (for console/dashboard)"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("JINNI LIVE ACCURACY METRICS")
        print("="*60)
        print(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%")
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Correct Predictions: {stats['correct_predictions']}")
        print(f"Average Error: {stats['average_error']:.2f}%")
        print(f"Improvement Rate: {stats['improvement_rate_per_second']:.4f}%/second")
        print(f"Running Time: {stats['elapsed_time_seconds']:.0f} seconds")
        print(f"Predictions/Second: {stats['predictions_per_second']:.2f}")
        print("="*60)
        
        # Top 5 stocks
        top_stocks = self.get_top_performing_stocks(5)
        if top_stocks:
            print("\nTop 5 Performing Stocks:")
            for symbol, data in top_stocks:
                print(f"  {symbol}: {data['accuracy']:.2f}% (Error: {data['avg_error']:.2f}%)")
        
        print("\n")
    
    def export_to_json(self, filename: str = 'accuracy_report.json'):
        """Export accuracy data to JSON"""
        report = {
            'statistics': self.get_statistics(),
            'stock_accuracies': self.stock_accuracies,
            'top_stocks': dict(self.get_top_performing_stocks(10)),
            'worst_stocks': dict(self.get_worst_performing_stocks(10)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        return filename
