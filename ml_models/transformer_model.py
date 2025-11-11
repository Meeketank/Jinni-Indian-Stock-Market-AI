import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import StandardScaler

class TransformerModel:
    """Transformer-based model for stock price prediction with attention mechanism"""
    
    def __init__(self, sequence_length=60, n_features=5, d_model=128, num_heads=8, 
                 dff=512, num_layers=4, dropout_rate=0.1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def positional_encoding(self, position, d_model):
        """Generate positional encodings for transformer"""
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = np.arange(position)[:, np.newaxis] * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def transformer_encoder(self, inputs, d_model, num_heads, dff, dropout_rate):
        """Single transformer encoder layer"""
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(inputs, inputs)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed forward network
        ffn_output = layers.Dense(dff, activation='relu')(out1)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2
    
    def build_model(self, input_shape):
        """Build the transformer model architecture"""
        inputs = layers.Input(shape=input_shape)
        
        # Project input to d_model dimension
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.sequence_length, self.d_model)
        x = x + pos_encoding[:, :self.sequence_length, :]
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Stack transformer encoder layers
        for _ in range(self.num_layers):
            x = self.transformer_encoder(x, self.d_model, self.num_heads, 
                                        self.dff, self.dropout_rate)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)  # Predict next price
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def prepare_data(self, data):
        """Prepare sequences for transformer"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, batch_size=32):
        """Train the transformer model"""
        X, y = self.prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        if self.model is None:
            self.model = self.build_model((X.shape[1], 1))
        
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                               validation_split=0.1, verbose=1)
        self.is_trained = True
        return history
    
    def predict(self, data):
        """Make predictions using transformer"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X_test = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        prediction = self.model.predict(X_test)
        return self.scaler.inverse_transform(prediction)
    
    def get_attention_weights(self):
        """Extract attention weights for visualization"""
        if self.model is None:
            return None
        attention_layers = [layer for layer in self.model.layers 
                          if isinstance(layer, layers.MultiHeadAttention)]
        return attention_layers
