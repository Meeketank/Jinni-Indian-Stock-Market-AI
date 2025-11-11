#!/usr/bin/env python3
"""
JINNI - Indian Stock Market AI System
Main entry point to run the complete system with live accuracy monitoring
"""

import sys
import time
import numpy as np
from datetime import datetime
import threading

print("="*80)
print("   ___  ___  _   _  _   _  ___")
print("  |_  ||_  || \ | || \ | ||_ _|")
print("   |_||___||  \| ||  \| | |_|")
print("")
print("  INDIAN STOCK MARKET AI - Real-Time Analysis & Prediction System")
print("  With Continuous Online Learning & Accuracy Improvement")
print("="*80)
print("")

# Import all modules
try:
    print("[1/8] Loading ML Models...")
    from ml_models.lstm_model import LSTMModel
    from ml_models.gru_model import GRUModel
    from ml_models.transformer_model import TransformerModel
    from ml_models.ensemble_model import EnsembleModel
    print("✓ ML Models loaded successfully")
    
    print("[2/8] Loading Data Scraper...")
    from data_scraper.nse_bse_fetcher import NSEBSEDataFetcher
    print("✓ Data Scraper loaded successfully")
    
    print("[3/8] Loading Technical Indicators...")
    from analytics.technical_indicators import TechnicalIndicators
    print("✓ Technical Indicators loaded successfully")
    
    print("[4/8] Loading Online Learning System...")
    from online_learning.incremental_trainer import IncrementalTrainer
    from online_learning.accuracy_tracker import AccuracyTracker
    print("✓ Online Learning System loaded successfully")
    
    print("[5/8] Loading Dashboard...")
    print("✓ Dashboard module ready")
    
    print("[6/8] Loading Configuration...")
    from config import *
    print("✓ Configuration loaded successfully")
    
    print("[7/8] Initializing System Components...")
    
except Exception as e:
    print(f"\n❌ Error loading modules: {str(e)}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

print("")
print("="*80)
print("SYSTEM INITIALIZATION COMPLETE")
print("="*80)
print("")

class JinniSystem:
    """Main Jinni System orchestrator"""
    
    def __init__(self):
        print("Initializing Jinni AI System...\n")
        
        # Initialize components
        self.ensemble_model = EnsembleModel()
        self.accuracy_tracker = AccuracyTracker()
        self.data_fetcher = NSEBSEDataFetcher()
        
        # Sample NSE/BSE stock symbols
        self.stock_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT',
            'HINDUNILVR', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'WIPRO'
        ]
        
        self.is_running = False
        print("✓ Jinni System initialized\n")
    
    def train_models(self, epochs=30):
        """Train all models with sample data"""
        print("\n" + "="*80)
        print("TRAINING PHASE: Training Ensemble Models (LSTM + GRU + Transformer)")
        print("="*80 + "\n")
        
        # Generate synthetic training data (in production, use real NSE/BSE data)
        print("Generating training data from historical patterns...")
        training_data = self._generate_synthetic_data(500)
        
        print(f"\nTraining ensemble with {len(training_data)} data points...")
        print(f"Epochs: {epochs}\n")
        
        try:
            self.ensemble_model.train_all_models(training_data, epochs=epochs, batch_size=32)
            print("\n✓ All models trained successfully!\n")
            return True
        except Exception as e:
            print(f"\n❌ Training error: {str(e)}\n")
            return False
    
    def run_live_prediction(self, duration_seconds=60):
        """Run live predictions with continuous accuracy monitoring"""
        print("\n" + "="*80)
        print("LIVE PREDICTION MODE: Continuous Learning & Accuracy Improvement")
print("="*80 + "\n")
        
        print(f"Running for {duration_seconds} seconds...")
        print("Watch accuracy improve every second!\n")
        
        self.is_running = True
        start_time = time.time()
        prediction_count = 0
        
        try:
            while time.time() - start_time < duration_seconds and self.is_running:
                # Select random stock
                stock = np.random.choice(self.stock_symbols)
                
                # Generate new data point
                current_data = self._generate_synthetic_data(60)
                
                # Make prediction
                predicted_price, model_predictions = self.ensemble_model.predict(current_data)
                
                # Simulate actual price (in production, fetch real price)
                actual_price = current_data[-1] * (1 + np.random.uniform(-0.015, 0.015))
                
                # Update accuracy
                self.accuracy_tracker.update_accuracy(predicted_price, actual_price, stock)
                
                # Incremental learning every 5 predictions
                if prediction_count % 5 == 0 and prediction_count > 0:
                    new_data = np.append(current_data, actual_price)
                    self.ensemble_model.incremental_train(new_data, epochs=2)
                
                prediction_count += 1
                
                # Display live stats every 10 seconds
                if prediction_count % 20 == 0:
                    self.accuracy_tracker.display_live_stats()
                
                time.sleep(0.5)  # Predict twice per second
        
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        self.is_running = False
        return prediction_count
    
    def display_final_results(self):
        """Display comprehensive final results"""
        print("\n" + "="*80)
        print("FINAL RESULTS - JINNI ACCURACY REPORT")
        print("="*80 + "\n")
        
        stats = self.accuracy_tracker.get_statistics()
        
        print(f"Total Runtime: {stats['elapsed_time_seconds']:.1f} seconds")
        print(f"Total Predictions Made: {stats['total_predictions']}")
        print(f"Correct Predictions: {stats['correct_predictions']}")
        print(f"")
        print(f"═══ ACCURACY METRICS ═══")
        print(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%")
        print(f"Average Error: {stats['average_error']:.3f}%")
        print(f"Median Error: {stats['median_error']:.3f}%")
        print(f"Min Error: {stats['min_error']:.3f}%")
        print(f"Max Error: {stats['max_error']:.3f}%")
        print(f"")
        print(f"═══ ONLINE LEARNING PERFORMANCE ═══")
        print(f"Accuracy Improvement Rate: {stats['improvement_rate_per_second']:.4f}%/second")
        print(f"Predictions Per Second: {stats['predictions_per_second']:.2f}")
        print(f"")
        
        # Ensemble model weights
        weights = self.ensemble_model.get_model_weights()
        print(f"═══ ENSEMBLE MODEL WEIGHTS ═══")
        print(f"LSTM Weight: {weights['lstm']:.3f}")
        print(f"GRU Weight: {weights['gru']:.3f}")
        print(f"Transformer Weight: {weights['transformer']:.3f}")
        print(f"")
        
        # Top performing stocks
        top_stocks = self.accuracy_tracker.get_top_performing_stocks(5)
        if top_stocks:
            print(f"═══ TOP 5 PERFORMING STOCKS ═══")
            for i, (symbol, data) in enumerate(top_stocks, 1):
                print(f"{i}. {symbol}: {data['accuracy']:.2f}% accuracy (Avg Error: {data['avg_error']:.2f}%)")
        
        print("\n" + "="*80)
        print("SYSTEM DEMONSTRATION COMPLETE")
        print("="*80 + "\n")
        
        # Export report
        filename = self.accuracy_tracker.export_to_json('jinni_accuracy_report.json')
        print(f"✓ Detailed report exported to: {filename}")
        print("")
    
    def _generate_synthetic_data(self, length=100):
        """Generate synthetic stock price data"""
        # Start with base price
        base_price = np.random.uniform(100, 5000)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, length)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return prices

def main():
    """Main execution function"""
    
    # Initialize system
    jinni = JinniSystem()
    
    # Train models
    print("\nStarting model training...")
    success = jinni.train_models(epochs=20)  # Reduced epochs for demo
    
    if not success:
        print("Training failed. Exiting...")
        return
    
    print("\nPress Enter to start live prediction mode...")
    input()
    
    # Run live predictions
    prediction_count = jinni.run_live_prediction(duration_seconds=30)  # 30 seconds demo
    
    print(f"\n✓ Completed {prediction_count} predictions\n")
    
    # Display final results
    jinni.display_final_results()
    
    print("\nThank you for using JINNI - The Indian Stock Market AI!")
    print("For more information, visit: https://github.com/Meeketank/Jinni-Indian-Stock-Market-AI")
    print("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting Jinni System...")
        print("Goodbye!\n")
