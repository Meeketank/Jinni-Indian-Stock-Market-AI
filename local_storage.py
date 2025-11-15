# local_storage.py
"""
Local SQLite storage manager - replaces Firebase dependency
Provides persistent storage for predictions, model weights, and metrics
"""

import sqlite3
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading

class LocalStorageManager:
    def __init__(self, db_path=".jinni_cache/jinni_data.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    predicted_price REAL,
                    actual_price REAL,
                    prediction_date TEXT,
                    target_date TEXT,
                    confidence REAL,
                    model_type TEXT,
                    horizon_days INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model weights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    weights_data BLOB,
                    metadata TEXT,
                    version INTEGER DEFAULT 1,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Recommendations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    recommendation TEXT,
                    confidence REAL,
                    reasons TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def log_prediction(self, symbol: str, predicted_price: float, 
                      confidence: float, horizon_days: int = 7,
                      model_type: str = "ensemble") -> int:
        """Log a prediction to database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            prediction_date = datetime.now().isoformat()
            from datetime import timedelta
            target_date = (datetime.now() + timedelta(days=horizon_days)).isoformat()
            
            cursor.execute('''
                INSERT INTO predictions 
                (symbol, predicted_price, confidence, horizon_days, 
                 model_type, prediction_date, target_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, predicted_price, confidence, horizon_days,
                  model_type, prediction_date, target_date))
            
            pred_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return pred_id
    
    def update_prediction_actual(self, prediction_id: int, actual_price: float):
        """Update prediction with actual price"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE predictions SET actual_price = ? WHERE id = ?
            ''', (actual_price, prediction_id))
            conn.commit()
            conn.close()
    
    def save_model_weights(self, model_name: str, weights: Any, metadata: Dict = None):
        """Save model weights to database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            weights_blob = pickle.dumps(weights)
            metadata_json = json.dumps(metadata or {})
            
            # Check if model exists
            cursor.execute('SELECT id FROM model_weights WHERE model_name = ?', (model_name,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE model_weights 
                    SET weights_data = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE model_name = ?
                ''', (weights_blob, metadata_json, model_name))
            else:
                cursor.execute('''
                    INSERT INTO model_weights (model_name, weights_data, metadata)
                    VALUES (?, ?, ?)
                ''', (model_name, weights_blob, metadata_json))
            
            conn.commit()
            conn.close()
    
    def load_model_weights(self, model_name: str) -> Optional[Any]:
        """Load model weights from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT weights_data FROM model_weights WHERE model_name = ? ORDER BY updated_at DESC LIMIT 1',
                (model_name,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return pickle.loads(result[0])
            return None
    
    def log_metric(self, metric_name: str, value: float, metadata: Dict = None):
        """Log a metric"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (metric_name, metric_value, metadata)
                VALUES (?, ?, ?)
            ''', (metric_name, value, json.dumps(metadata or {})))
            conn.commit()
            conn.close()
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[Dict]:
        """Get recent metrics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = cursor.execute('''
                SELECT metric_value, metadata, recorded_at 
                FROM metrics 
                WHERE metric_name = ?
                ORDER BY recorded_at DESC LIMIT ?
            ''', (metric_name, limit))
            results = cursor.fetchall()
            conn.close()
            
            return [{
                'value': r[0],
                'metadata': json.loads(r[1]) if r[1] else {},
                'timestamp': r[2]
            } for r in results]
    
    def get_prediction_accuracy(self, days_back: int = 30) -> Dict:
        """Calculate prediction accuracy"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT predicted_price, actual_price, confidence
                FROM predictions
                WHERE actual_price IS NOT NULL
                  AND created_at >= datetime('now', '-' || ? || ' days')
            ''', (days_back,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'accuracy': 0.0, 'mae': 0.0, 'samples': 0}
            
            import numpy as np
            preds = np.array([r[0] for r in results])
            actuals = np.array([r[1] for r in results])
            
            mae = float(np.mean(np.abs(preds - actuals)))
            direction_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
            
            return {
                'accuracy': direction_acc * 100,
                'mae': mae,
                'samples': len(results)
            }
    
    def save_recommendation(self, symbol: str, recommendation: str, 
                           confidence: float, reasons: List[str]):
        """Save recommendation"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO recommendations (symbol, recommendation, confidence, reasons)
                VALUES (?, ?, ?, ?)
            ''', (symbol, recommendation, confidence, json.dumps(reasons)))
            conn.commit()
            conn.close()
    
    def get_recent_predictions(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get recent predictions"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                query = '''
                    SELECT * FROM predictions 
                    WHERE symbol = ?
                    ORDER BY created_at DESC LIMIT ?
                '''
                cursor.execute(query, (symbol, limit))
            else:
                query = '''
                    SELECT * FROM predictions 
                    ORDER BY created_at DESC LIMIT ?
                '''
                cursor.execute(query, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return results

# Compatibility wrapper for Firebase-like interface
class StorageManager(LocalStorageManager):
    """Alias for compatibility"""
    pass
