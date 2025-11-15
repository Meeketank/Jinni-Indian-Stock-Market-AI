# firebase_config.py
"""
Firebase Configuration and Integration Module for JINNI
Handles persistent storage of:
- Predictions and actual outcomes
- Model training data
- Scan results
- Performance metrics
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase libraries not installed. Running in offline mode.")

class FirebaseManager:
    def __init__(self):
        self.db = None
        self.initialized = False
        
        if not FIREBASE_AVAILABLE:
            print("Firebase not available - using local storage fallback")
            return
            
        try:
            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                # Try to use service account from Streamlit secrets
                if hasattr(st, 'secrets') and 'firebase' in st.secrets:
                    cred_dict = dict(st.secrets['firebase'])
                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                else:
                    # Fallback: try environment variable or local file
                    cred_path = os.getenv('FIREBASE_CREDENTIALS', 'firebase-credentials.json')
                    if os.path.exists(cred_path):
                        cred = credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred)
                    else:
                        # Use application default credentials
                        firebase_admin.initialize_app()
            
            self.db = firestore.client()
            self.initialized = True
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            self.initialized = False
    
    def is_available(self) -> bool:
        return self.initialized and self.db is not None
    
    # ===== PREDICTION STORAGE =====
    def log_prediction(self, symbol: str, prediction_data: Dict[str, Any]) -> Optional[str]:
        """
        Log a prediction to Firebase for later validation
        Returns document ID if successful
        """
        if not self.is_available():
            return None
        
        try:
            doc_data = {
                'symbol': symbol.upper(),
                'predicted_price': prediction_data.get('predicted_price'),
                'current_price': prediction_data.get('current_price'),
                'predicted_return_pct': prediction_data.get('predicted_return_pct'),
                'confidence': prediction_data.get('confidence'),
                'horizon_days': prediction_data.get('horizon_days'),
                'model_type': prediction_data.get('model_type', 'ensemble'),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'validation_date': (datetime.now() + timedelta(days=prediction_data.get('horizon_days', 7))).isoformat(),
                'actual_outcome': None,  # To be filled later
                'validated': False
            }
            
            doc_ref = self.db.collection('predictions').add(doc_data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error logging prediction: {e}")
            return None
    
    def update_prediction_outcome(self, doc_id: str, actual_price: float, actual_return_pct: float):
        """
        Update a prediction with actual outcome for learning
        """
        if not self.is_available():
            return
        
        try:
            self.db.collection('predictions').document(doc_id).update({
                'actual_price': actual_price,
                'actual_return_pct': actual_return_pct,
                'validated': True,
                'validation_timestamp': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"Error updating prediction outcome: {e}")
    
    def get_unvalidated_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get predictions that need validation (validation_date has passed)
        """
        if not self.is_available():
            return []
        
        try:
            now = datetime.now().isoformat()
            docs = self.db.collection('predictions')\
                .where('validated', '==', False)\
                .where('validation_date', '<=', now)\
                .limit(limit)\
                .stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                data['doc_id'] = doc.id
                results.append(data)
            return results
        except Exception as e:
            print(f"Error getting unvalidated predictions: {e}")
            return []
    
    # ===== SCAN RESULTS STORAGE =====
    def save_scan_result(self, scan_data: Dict[str, Any]) -> Optional[str]:
        """
        Save scan & recommendation results
        """
        if not self.is_available():
            return None
        
        try:
            doc_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'scan_params': {
                    'min_expected_pct': scan_data.get('min_expected_pct'),
                    'horizon_days': scan_data.get('horizon_days'),
                    'symbols_scanned': scan_data.get('symbols_scanned'),
                    'scan_all': scan_data.get('scan_all', False)
                },
                'recommendations': scan_data.get('recommendations', []),
                'model_metrics': scan_data.get('model_metrics', {})
            }
            
            doc_ref = self.db.collection('scan_results').add(doc_data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving scan result: {e}")
            return None
    
    def get_recent_scans(self, limit: int = 20) -> List[Dict]:
        """
        Retrieve recent scan results
        """
        if not self.is_available():
            return []
        
        try:
            docs = self.db.collection('scan_results')\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                data['doc_id'] = doc.id
                results.append(data)
            return results
        except Exception as e:
            print(f"Error getting recent scans: {e}")
            return []
    
    # ===== TRAINING DATA STORAGE =====
    def save_training_batch(self, batch_data: Dict[str, Any]) -> Optional[str]:
        """
        Save a batch of training data for model improvement
        """
        if not self.is_available():
            return None
        
        try:
            doc_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'symbols': batch_data.get('symbols', []),
                'samples_count': batch_data.get('samples_count'),
                'model_type': batch_data.get('model_type'),
                'metrics_before': batch_data.get('metrics_before'),
                'metrics_after': batch_data.get('metrics_after')
            }
            
            doc_ref = self.db.collection('training_history').add(doc_data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving training batch: {e}")
            return None
    
    # ===== MODEL METRICS STORAGE =====
    def save_model_metrics(self, metrics: Dict[str, Any]):
        """
        Save current model performance metrics
        """
        if not self.is_available():
            return
        
        try:
            doc_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'directional_accuracy': metrics.get('directional_accuracy'),
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'trained_samples': metrics.get('trained_samples'),
                'model_version': metrics.get('model_version', '1.0')
            }
            
            self.db.collection('model_metrics').add(doc_data)
        except Exception as e:
            print(f"Error saving model metrics: {e}")
    
    def get_metrics_history(self, days: int = 30) -> List[Dict]:
        """
        Get model performance history
        """
        if not self.is_available():
            return []
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            docs = self.db.collection('model_metrics')\
                .where('timestamp', '>=', cutoff)\
                .order_by('timestamp')\
                .stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                results.append(data)
            return results
        except Exception as e:
            print(f"Error getting metrics history: {e}")
            return []
    
    # ===== ANALYTICS =====
    def get_prediction_accuracy_stats(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate accuracy statistics from validated predictions
        """
        if not self.is_available():
            return {}
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            docs = self.db.collection('predictions')\
                .where('validated', '==', True)\
                .where('validation_timestamp', '>=', cutoff)\
                .stream()
            
            correct_direction = 0
            total = 0
            abs_errors = []
            
            for doc in docs:
                data = doc.to_dict()
                pred_return = data.get('predicted_return_pct', 0)
                actual_return = data.get('actual_return_pct', 0)
                
                if pred_return * actual_return > 0:  # Same sign
                    correct_direction += 1
                
                abs_errors.append(abs(pred_return - actual_return))
                total += 1
            
            if total == 0:
                return {'directional_accuracy': 0, 'mae': 0, 'total_predictions': 0}
            
            return {
                'directional_accuracy': correct_direction / total,
                'mae': sum(abs_errors) / len(abs_errors),
                'total_predictions': total
            }
        except Exception as e:
            print(f"Error calculating prediction accuracy: {e}")
            return {}

# Global instance
_firebase_manager = None

def get_firebase_manager() -> FirebaseManager:
    """Get or create Firebase manager singleton"""
    global _firebase_manager
    if _firebase_manager is None:
        _firebase_manager = FirebaseManager()
    return _firebase_manager
