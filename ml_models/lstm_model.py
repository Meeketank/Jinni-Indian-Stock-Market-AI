# LSTM Model for Stock Price Prediction
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    def __init__(self, sequence_length=60, units=128):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(self.units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.units//2),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, batch_size=32):
        X, y = self.prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        if self.model is None:
            self.build_model((X.shape[1], 1))
        
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                validation_split=0.1, verbose=1)
        self.is_trained = True
        return history
    
    def predict(self, data):
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X_test = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        prediction = self.model.predict(X_test)
        return self.scaler.inverse_transform(prediction)[0][0]
    
    def save_model(self, path):
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.is_trained = True
