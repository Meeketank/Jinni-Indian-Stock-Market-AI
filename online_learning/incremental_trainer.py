# Online Learning - Incremental Training System
import numpy as np
import logging
from datetime import datetime
from river import linear_model, preprocessing, metrics

logger = logging.getLogger(__name__)

class IncrementalTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics_tracker = {}
        self.learning_rate = 0.01
        logger.info("Incremental Trainer initialized")
    
    def initialize_model(self, symbol):
        self.models[symbol] = linear_model.LinearRegression()
        self.scalers[symbol] = preprocessing.StandardScaler()
        self.metrics_tracker[symbol] = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE(),
            'samples_processed': 0,
            'last_updated': datetime.now()
        }
        logger.info(f"Model initialized for {symbol}")
    
    def update_model(self, symbol, features, target):
        if symbol not in self.models:
            self.initialize_model(symbol)
        
        try:
            X_scaled = {}
            for key, value in features.items():
                X_scaled[key] = self.scalers[symbol].learn_one({key: value}).transform_one({key: value})[key]
            
            prediction = self.models[symbol].predict_one(X_scaled)
            
            self.models[symbol].learn_one(X_scaled, target)
            
            self.metrics_tracker[symbol]['mae'].update(target, prediction)
            self.metrics_tracker[symbol]['rmse'].update(target, prediction)
            self.metrics_tracker[symbol]['samples_processed'] += 1
            self.metrics_tracker[symbol]['last_updated'] = datetime.now()
            
            return prediction
        except Exception as e:
            logger.error(f"Error updating model for {symbol}: {e}")
            return None
    
    def get_accuracy(self, symbol):
        if symbol in self.metrics_tracker:
            return {
                'mae': self.metrics_tracker[symbol]['mae'].get(),
                'rmse': self.metrics_tracker[symbol]['rmse'].get(),
                'samples': self.metrics_tracker[symbol]['samples_processed']
            }
        return None
    
    def predict(self, symbol, features):
        if symbol not in self.models:
            return None
        
        try:
            X_scaled = {}
            for key, value in features.items():
                X_scaled[key] = self.scalers[symbol].transform_one({key: value})[key]
            
            return self.models[symbol].predict_one(X_scaled)
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return None
    
    def get_all_accuracies(self):
        accuracies = {}
        for symbol in self.metrics_tracker:
            accuracies[symbol] = self.get_accuracy(symbol)
        return accuracies
