import os
import sys
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_models import AdvancedEnsembleModel
    from automated_retraining import AutomatedRetrainingSystem
    from local_storage import StorageManager
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearner:
    """
    Autonomous learning system that learns from market behavior continuously.
    Doesn't require app restarts - learns persist across sessions.
    Integrates with AdvancedEnsembleModel and AutomatedRetrainingSystem.
    """
    
    def __init__(self, checkpoint_interval=3600):
        """Initialize continuous learner with persistence"""
        self.logger = logger
        self.logger.info("Initializing Continuous Learner...")
        
        self.storage = StorageManager()
        self.ensemble_model = AdvancedEnsembleModel()
        self.retraining_system = AutomatedRetrainingSystem()
        
        # Persistence settings
        self.checkpoint_interval = checkpoint_interval  # Save every hour
        self.last_checkpoint = datetime.now()
        self.learning_metrics = self._load_learning_metrics()
        
        # Learning state
        self.predictions_since_last_save = 0
        self.accuracies = []  
        self.losses = []
        self.model_versions = []
        
        self.running = False
        self.stop_event = threading.Event()
        
        self.logger.info("Continuous Learner initialized")
    
    def _load_learning_metrics(self):
        """Load previously learned metrics from storage"""
        try:
            metrics = self.storage.get_learning_metrics()
            if metrics:
                self.logger.info(f"Loaded learning metrics: {len(metrics)} records")
                return metrics
        except:
            pass
        return {}
    
    def _save_checkpoint(self):
        """Save learning checkpoint to persist across sessions"""
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'accuracies': self.accuracies[-100:],  # Keep last 100
                'losses': self.losses[-100:],
                'model_versions': len(self.model_versions),
                'predictions_processed': self.predictions_since_last_save,
                'metrics': self.learning_metrics
            }
            
            self.storage.save_checkpoint(checkpoint)
            self.predictions_since_last_save = 0
            self.logger.info(f"Checkpoint saved: accuracies={len(self.accuracies)}, losses={len(self.losses)}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load previous checkpoint if available"""
        try:
            checkpoint = self.storage.get_latest_checkpoint()
            if checkpoint:
                self.accuracies = checkpoint.get('accuracies', [])
                self.losses = checkpoint.get('losses', [])
                self.model_versions = [None] * checkpoint.get('model_versions', 0)
                self.logger.info(f"Loaded checkpoint: {len(self.accuracies)} accuracy records")
                return True
        except:
            pass
        return False
    
    def learn_from_market(self):
        """
        Continuous learning loop that learns from market movements.
        Runs indefinitely without stopping.
        """
        self.logger.info("Starting continuous market learning...")
        self.running = True
        self._load_checkpoint()
        
        while not self.stop_event.is_set():
            try:
                # Get recent predictions and actual prices
                recent_data = self.storage.get_recent_predictions(limit=200)
                
                if recent_data and len(recent_data) > 10:
                    # Learn from price movements
                    learning_success = self._learn_from_predictions(recent_data)
                    
                    if learning_success:
                        self.predictions_since_last_save += len(recent_data)
                    
                    # Save checkpoint periodically
                    if (datetime.now() - self.last_checkpoint).seconds > self.checkpoint_interval:
                        self._save_checkpoint()
                        self.last_checkpoint = datetime.now()
                
                time.sleep(300)  # Learn every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
                time.sleep(600)
    
    def _learn_from_predictions(self, predictions):
        """
        Learn from prediction accuracy and market behavior.
        Update models based on prediction errors.
        """
        try:
            accuracy_improvements = []
            
            for pred in predictions:
                try:
                    symbol = pred.get('symbol')
                    prediction = pred.get('prediction', {})
                    confidence = prediction.get('confidence', 0)
                    
                    # Get actual price movement
                    actual_data = self.storage.get_actual_price_movement(symbol)
                    
                    if actual_data:
                        predicted_direction = prediction.get('direction')  # 'UP' or 'DOWN'
                        actual_direction = actual_data.get('direction')
                        
                        # Calculate accuracy
                        if predicted_direction and actual_direction:
                            is_correct = predicted_direction == actual_direction
                            accuracy = 1.0 if is_correct else 0.0
                            self.accuracies.append(accuracy)
                            
                            # Calculate loss
                            loss = abs(prediction.get('price_target', 0) - actual_data.get('price', 0))
                            self.losses.append(loss)
                            
                            accuracy_improvements.append({
                                'symbol': symbol,
                                'accuracy': accuracy,
                                'confidence_before': confidence
                            })
                            
                except Exception as e:
                    self.logger.debug(f"Error learning from {symbol}: {e}")
                    continue
            
            if accuracy_improvements:
                # Update model weights based on performance
                avg_accuracy = np.mean([x['accuracy'] for x in accuracy_improvements])
                self.logger.info(f"Learning update: avg_accuracy={avg_accuracy:.3f}, samples={len(accuracy_improvements)}")
                
                # Store learning metrics
                self.storage.save_learning_metrics({
                    'timestamp': datetime.now().isoformat(),
                    'avg_accuracy': avg_accuracy,
                    'samples': len(accuracy_improvements),
                    'improvements': accuracy_improvements
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in learning: {e}")
            return False
    
    def trigger_model_update(self):
        """
        Trigger model update based on accumulated learning.
        Updates ensemble model with new insights.
        """
        try:
            if len(self.accuracies) < 10:
                self.logger.info("Not enough data for model update")
                return False
            
            avg_accuracy = np.mean(self.accuracies[-50:])
            self.logger.info(f"Triggering model update: avg_accuracy={avg_accuracy:.3f}")
            
            # Update ensemble model
            self.ensemble_model.update_weights(self.accuracies[-50:])
            
            # Run retraining if needed
            if avg_accuracy < 0.6:  # If accuracy drops below 60%
                self.logger.info("Accuracy below threshold - triggering full retraining")
                recent_data = self.storage.get_recent_predictions(limit=500)
                self.retraining_system.run_retraining(recent_data)
            
            self.model_versions.append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': avg_accuracy
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False
    
    def get_learning_status(self):
        """Get current learning status and metrics"""
        if len(self.accuracies) == 0:
            return {
                'status': 'initializing',
                'predictions_learned': 0,
                'avg_accuracy': 0
            }
        
        return {
            'status': 'learning',
            'predictions_learned': len(self.accuracies),
            'avg_accuracy': np.mean(self.accuracies[-100:]),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'model_versions': len(self.model_versions),
            'predictions_since_checkpoint': self.predictions_since_last_save
        }
    
    def start(self):
        """Start continuous learning in background thread"""
        if not self.running:
            self.stop_event.clear()
            learner_thread = threading.Thread(target=self.learn_from_market, daemon=True)
            learner_thread.start()
            self.logger.info("Continuous learner started")
    
    def stop(self):
        """Stop learning and save final checkpoint"""
        self._save_checkpoint()
        self.stop_event.set()
        self.running = False
        self.logger.info("Continuous learner stopped")


if __name__ == "__main__":
    learner = ContinuousLearner()
    learner.start()
    
    try:
        while True:
            status = learner.get_learning_status()
            print(f"Learning Status: {status}")
            time.sleep(60)
    except KeyboardInterrupt:
        learner.stop()
        print("\nLearner shut down")
