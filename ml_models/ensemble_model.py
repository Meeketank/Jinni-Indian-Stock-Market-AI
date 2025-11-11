import numpy as np
from sklearn.preprocessing import StandardScaler
from ml_models.lstm_model import LSTMModel
from ml_models.gru_model import GRUModel
from ml_models.transformer_model import TransformerModel

class EnsembleModel:
    """Ensemble model combining LSTM, GRU, and Transformer predictions"""
    
    def __init__(self, sequence_length=60, weights=None):
        self.sequence_length = sequence_length
        self.lstm_model = LSTMModel(sequence_length=sequence_length)
        self.gru_model = GRUModel(sequence_length=sequence_length)
        self.transformer_model = TransformerModel(sequence_length=sequence_length)
        
        # Default equal weights if not provided
        self.weights = weights if weights else {'lstm': 0.33, 'gru': 0.33, 'transformer': 0.34}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_accuracies = {'lstm': 0.0, 'gru': 0.0, 'transformer': 0.0}
    
    def train_all_models(self, data, epochs=50, batch_size=32):
        """Train all component models"""
        print("Training LSTM model...")
        lstm_history = self.lstm_model.train(data, epochs=epochs, batch_size=batch_size)
        
        print("\nTraining GRU model...")
        gru_history = self.gru_model.train(data, epochs=epochs, batch_size=batch_size)
        
        print("\nTraining Transformer model...")
        transformer_history = self.transformer_model.train(data, epochs=epochs, batch_size=batch_size)
        
        self.is_trained = True
        
        return {
            'lstm': lstm_history,
            'gru': gru_history,
            'transformer': transformer_history
        }
    
    def update_weights_dynamically(self, validation_data):
        """Update ensemble weights based on individual model performance"""
        if not self.is_trained:
            raise ValueError("Models must be trained before updating weights")
        
        # Calculate accuracy for each model
        lstm_pred = self.lstm_model.predict(validation_data)
        gru_pred = self.gru_model.predict(validation_data)
        transformer_pred = self.transformer_model.predict(validation_data)
        
        # Calculate errors (lower is better)
        actual = validation_data[-1]
        lstm_error = abs(lstm_pred[0][0] - actual)
        gru_error = abs(gru_pred[0][0] - actual)
        transformer_error = abs(transformer_pred[0][0] - actual)
        
        # Convert errors to accuracies (inverse relationship)
        total_error = lstm_error + gru_error + transformer_error
        if total_error > 0:
            self.weights['lstm'] = 1 - (lstm_error / total_error)
            self.weights['gru'] = 1 - (gru_error / total_error)
            self.weights['transformer'] = 1 - (transformer_error / total_error)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Store accuracies for reporting
        self.model_accuracies = {
            'lstm': self.weights['lstm'],
            'gru': self.weights['gru'],
            'transformer': self.weights['transformer']
        }
        
        return self.weights
    
    def predict(self, data):
        """Make ensemble prediction using weighted average"""
        if not self.is_trained:
            raise ValueError("Models not trained")
        
        # Get predictions from each model
        lstm_pred = self.lstm_model.predict(data)
        gru_pred = self.gru_model.predict(data)
        transformer_pred = self.transformer_model.predict(data)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.weights['lstm'] * lstm_pred[0][0] +
            self.weights['gru'] * gru_pred[0][0] +
            self.weights['transformer'] * transformer_pred[0][0]
        )
        
        return ensemble_pred, {
            'lstm': lstm_pred[0][0],
            'gru': gru_pred[0][0],
            'transformer': transformer_pred[0][0],
            'ensemble': ensemble_pred
        }
    
    def predict_multiple_steps(self, data, steps=5):
        """Predict multiple steps ahead"""
        predictions = []
        current_data = data.copy()
        
        for step in range(steps):
            pred, _ = self.predict(current_data)
            predictions.append(pred)
            
            # Update data with prediction for next step
            current_data = np.append(current_data[1:], pred)
        
        return predictions
    
    def get_model_weights(self):
        """Return current ensemble weights"""
        return self.weights
    
    def get_model_accuracies(self):
        """Return individual model accuracies"""
        return self.model_accuracies
    
    def incremental_train(self, new_data, epochs=5):
        """Incrementally train all models with new data"""
        print("Incremental training LSTM...")
        self.lstm_model.train(new_data, epochs=epochs, batch_size=16)
        
        print("Incremental training GRU...")
        self.gru_model.train(new_data, epochs=epochs, batch_size=16)
        
        print("Incremental training Transformer...")
        self.transformer_model.train(new_data, epochs=epochs, batch_size=16)
        
        print("Updating ensemble weights...")
        self.update_weights_dynamically(new_data)
    
    def get_ensemble_summary(self):
        """Get summary of ensemble performance"""
        return {
            'is_trained': self.is_trained,
            'weights': self.weights,
            'accuracies': self.model_accuracies,
            'sequence_length': self.sequence_length,
            'models': ['LSTM', 'GRU', 'Transformer']
        }
