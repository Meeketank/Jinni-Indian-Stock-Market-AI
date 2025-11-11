import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class XGBoostModel:
    """XGBoost model for high-accuracy stock predictions"""
    
    def __init__(self, sequence_length=60, n_estimators=1000, learning_rate=0.01, max_depth=7):
        self.sequence_length = sequence_length
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # XGBoost parameters for optimal performance
        self.params = {
            'objective': 'reg:squarederror',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def prepare_data(self, data, create_features=True):
        """Prepare sequences with engineered features"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            sequence = scaled_data[i-self.sequence_length:i, 0]
            
            if create_features:
                # Add engineered features
                features = self._engineer_features(sequence)
                X.append(features)
            else:
                X.append(sequence)
            
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def _engineer_features(self, sequence):
        """Create advanced features from sequence"""
        features = list(sequence)
        
        # Statistical features
        features.append(np.mean(sequence))  # Mean
        features.append(np.std(sequence))   # Standard deviation
        features.append(np.min(sequence))   # Min
        features.append(np.max(sequence))   # Max
        features.append(sequence[-1] - sequence[0])  # Total change
        
        # Momentum features
        if len(sequence) > 1:
            features.append(sequence[-1] - sequence[-2])  # Last change
            features.append(np.mean(np.diff(sequence)))   # Avg change
        
        # Moving averages
        if len(sequence) >= 5:
            features.append(np.mean(sequence[-5:]))  # MA5
        if len(sequence) >= 10:
            features.append(np.mean(sequence[-10:]))  # MA10
        if len(sequence) >= 20:
            features.append(np.mean(sequence[-20:]))  # MA20
        
        # Trend features
        if len(sequence) > 2:
            # Simple linear regression slope
            x = np.arange(len(sequence))
            slope = np.polyfit(x, sequence, 1)[0]
            features.append(slope)
        
        return features
    
    def train(self, data, epochs=None, batch_size=None):
        """Train XGBoost model"""
        X, y = self.prepare_data(data, create_features=True)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Calculate training accuracy
        predictions = self.model.predict(dtrain)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy': 100 * (1 - mae)  # Simplified accuracy metric
        }
    
    def predict(self, data):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Prepare last sequence
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        sequence = scaled_data[-self.sequence_length:, 0]
        
        # Engineer features
        features = self._engineer_features(sequence)
        X_test = np.array([features])
        
        # Predict
        dtest = xgb.DMatrix(X_test)
        prediction = self.model.predict(dtest)
        
        # Inverse transform
        prediction_reshaped = prediction.reshape(-1, 1)
        original_scale = self.scaler.inverse_transform(prediction_reshaped)
        
        return original_scale
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            return None
        
        importance = self.model.get_score(importance_type='weight')
        return importance
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(filepath)
        
        # Save scaler separately
        with open(filepath + '.scaler', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        
        # Load scaler
        with open(filepath + '.scaler', 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True
    
    def incremental_train(self, new_data, epochs=100):
        """Incrementally train with new data"""
        X, y = self.prepare_data(new_data, create_features=True)
        
        # Continue training
        dtrain = xgb.DMatrix(X, label=y)
        
        if self.model is not None:
            # Update existing model
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=epochs,
                xgb_model=self.model,
                verbose_eval=False
            )
        else:
            # Train new model
            self.train(new_data)
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        X, y = self.prepare_data(test_data, create_features=True)
        dtest = xgb.DMatrix(X)
        
        predictions = self.model.predict(dtest)
        
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'accuracy': 100 * (1 - mae)
        }
