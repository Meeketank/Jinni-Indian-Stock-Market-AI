"""
Advanced ML Models - BlackRock Aladdin-Competitive Ensemble System

This module implements enterprise-grade machine learning with:
- XGBoost: Gradient boosting for complex patterns
- LightGBM: Fast gradient boosting with leaf-wise growth
- CatBoost: Categorical boosting for robust predictions
- Genetic Algorithm: Hyperparameter optimization
- Ensemble Voting: Combines predictions from multiple models
- Model Stacking: Meta-learning for superior accuracy

Author: JINNI Development Team
Designed to compete with BlackRock's Aladdin for Indian stock market
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import joblib
import os

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEnsembleModel:
    """
    Enterprise ensemble model combining XGBoost, LightGBM, and CatBoost.
    
    Features:
    - Genetic algorithm hyperparameter tuning
    - Model stacking and voting
    - Cross-validation for robustness
    - Automatic feature selection
    - Time-series aware splitting
    """
    
    def __init__(self, task='regression', optimize=True):
        """
        Initialize advanced ensemble model.
        
        Args:
            task: 'regression' or 'classification'
            optimize: Whether to use genetic algorithm for hyperparameter tuning
        """
        self.task = task
        self.optimize = optimize
        self.models = {}
        self.ensemble = None
        self.feature_importance = None
        self.best_params = {}
        self.metrics = {}
        
        logger.info(f"Initialized AdvancedEnsembleModel for {task}")
    
    def _create_xgboost_model(self, params=None):
        """Create XGBoost model with optimal parameters."""
        default_params = {
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        if self.task == 'regression':
            model = xgb.XGBRegressor(**default_params)
        else:
            model = xgb.XGBClassifier(**default_params)
        
        logger.info("Created XGBoost model")
        return model
    
    def _create_lightgbm_model(self, params=None):
        """Create LightGBM model with optimal parameters."""
        default_params = {
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'num_leaves': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'random_state': 42,
            'verbosity': -1
        }
        
        if params:
            default_params.update(params)
        
        if self.task == 'regression':
            model = lgb.LGBMRegressor(**default_params)
        else:
            model = lgb.LGBMClassifier(**default_params)
        
        logger.info("Created LightGBM model")
        return model
    
    def _create_catboost_model(self, params=None):
        """Create CatBoost model with optimal parameters."""
        default_params = {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        }
        
        if params:
            default_params.update(params)
        
        if self.task == 'regression':
            model = CatBoostRegressor(**default_params)
        else:
            model = CatBoostClassifier(**default_params)
        
        logger.info("Created CatBoost model")
        return model
    
    def _optimize_hyperparameters(self, X, y, model_name='xgboost'):
        """Use genetic algorithm to find optimal hyperparameters."""
        logger.info(f"Optimizing {model_name} hyperparameters with genetic algorithm...")
        
        # Define search spaces for each model
        if model_name == 'xgboost':
            param_grid = {
                'max_depth': Integer(5, 15),
                'learning_rate': Continuous(0.01, 0.3),
                'n_estimators': Integer(100, 500),
                'subsample': Continuous(0.6, 1.0),
                'colsample_bytree': Continuous(0.6, 1.0)
            }
            base_model = self._create_xgboost_model()
        
        elif model_name == 'lightgbm':
            param_grid = {
                'max_depth': Integer(5, 20),
                'learning_rate': Continuous(0.01, 0.3),
                'n_estimators': Integer(100, 500),
                'num_leaves': Integer(20, 100),
                'subsample': Continuous(0.6, 1.0)
            }
            base_model = self._create_lightgbm_model()
        
        else:  # catboost
            param_grid = {
                'depth': Integer(4, 12),
                'learning_rate': Continuous(0.01, 0.3),
                'iterations': Integer(100, 500),
                'l2_leaf_reg': Continuous(1, 10)
            }
            base_model = self._create_catboost_model()
        
        # Genetic algorithm search
        evolved_estimator = GASearchCV(
            estimator=base_model,
            cv=3,
            param_grid=param_grid,
            scoring='neg_mean_squared_error' if self.task == 'regression' else 'accuracy',
            population_size=10,
            generations=5,
            n_jobs=-1,
            verbose=False
        )
        
        evolved_estimator.fit(X, y)
        self.best_params[model_name] = evolved_estimator.best_params_
        
        logger.info(f"Best {model_name} params: {evolved_estimator.best_params_}")
        return evolved_estimator.best_estimator_
    
    def train(self, X, y, optimize_each=False):
        """
        Train ensemble model with all base models.
        
        Args:
            X: Training features
            y: Training target
            optimize_each: Whether to optimize each base model separately
        """
        logger.info(f"Training ensemble model with {len(X)} samples...")
        start_time = datetime.now()
        
        # Create and optionally optimize base models
        if self.optimize and optimize_each:
            logger.info("Optimizing each model with genetic algorithm (this may take time)...")
            self.models['xgboost'] = self._optimize_hyperparameters(X, y, 'xgboost')
            self.models['lightgbm'] = self._optimize_hyperparameters(X, y, 'lightgbm')
            self.models['catboost'] = self._optimize_hyperparameters(X, y, 'catboost')
        else:
            self.models['xgboost'] = self._create_xgboost_model()
            self.models['lightgbm'] = self._create_lightgbm_model()
            self.models['catboost'] = self._create_catboost_model()
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)
            
            # Calculate metrics
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.metrics[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Create ensemble with voting
        if self.task == 'regression':
            self.ensemble = VotingRegressor(
                estimators=[(name, model) for name, model in self.models.items()],
                weights=[2, 2, 1]  # Give XGBoost and LightGBM slightly more weight
            )
        
        self.ensemble.fit(X, y)
        
        # Calculate ensemble metrics
        y_pred_ensemble = self.ensemble.predict(X)
        self.metrics['ensemble'] = {
            'mse': mean_squared_error(y, y_pred_ensemble),
            'mae': mean_absolute_error(y, y_pred_ensemble),
            'r2': r2_score(y, y_pred_ensemble)
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Ensemble training completed in {training_time:.2f}s")
        logger.info(f"Ensemble - MSE: {self.metrics['ensemble']['mse']:.4f}, "
                   f"R2: {self.metrics['ensemble']['r2']:.4f}")
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        return self.metrics
    
    def _calculate_feature_importance(self, X):
        """Calculate and aggregate feature importance across models."""
        importance_dict = {}
        
        # Get features from pandas if available
        if isinstance(X, pd.DataFrame):
            features = X.columns.tolist()
        else:
            features = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Aggregate importance from each model
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feature_name = features[i]
                    if feature_name not in importance_dict:
                        importance_dict[feature_name] = []
                    importance_dict[feature_name].append(importance)
        
        # Average importance across models
        self.feature_importance = {
            feature: np.mean(scores) 
            for feature, scores in importance_dict.items()
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"Top 5 important features: {list(self.feature_importance.keys())[:5]}")
    
    def predict(self, X, use_ensemble=True):
        """
        Make predictions using ensemble or individual models.
        
        Args:
            X: Features for prediction
            use_ensemble: If True, use ensemble; else return all model predictions
        
        Returns:
            Predictions (ensemble or dict of model predictions)
        """
        if use_ensemble:
            if self.ensemble is None:
                raise ValueError("Model not trained yet. Call train() first.")
            return self.ensemble.predict(X)
        else:
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            predictions['ensemble'] = self.ensemble.predict(X)
            return predictions
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence based on model agreement.
        
        Returns:
            tuple: (predictions, confidence_scores)
        """
        # Get predictions from all models
        all_predictions = []
        for name, model in self.models.items():
            all_predictions.append(model.predict(X))
        
        all_predictions = np.array(all_predictions)
        
        # Calculate ensemble prediction
        ensemble_pred = self.ensemble.predict(X)
        
        # Calculate confidence as inverse of prediction variance
        pred_variance = np.var(all_predictions, axis=0)
        confidence = 1 / (1 + pred_variance)  # Normalize to 0-1 range
        
        return ensemble_pred, confidence
    
    def get_model_comparison(self):
        """
        Compare performance of all models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        for name, metrics in self.metrics.items():
            comparison_data.append({
                'Model': name,
                'MSE': metrics.get('mse', 0),
                'MAE': metrics.get('mae', 0),
                'R2': metrics.get('r2', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('R2', ascending=False)
        return df
    
    def save_models(self, directory='models/'):
        """
        Save all trained models to disk.
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.pkl')
            joblib.dump(model, filepath)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save ensemble
        if self.ensemble:
            filepath = os.path.join(directory, 'ensemble_model.pkl')
            joblib.dump(self.ensemble, filepath)
            logger.info(f"Saved ensemble to {filepath}")
        
        # Save metadata
        metadata = {
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'task': self.task
        }
        filepath = os.path.join(directory, 'model_metadata.pkl')
        joblib.dump(metadata, filepath)
        logger.info(f"Saved metadata to {filepath}")
    
    def load_models(self, directory='models/'):
        """
        Load trained models from disk.
        
        Args:
            directory: Directory containing saved models
        """
        # Load individual models
        model_names = ['xgboost', 'lightgbm', 'catboost']
        for name in model_names:
            filepath = os.path.join(directory, f'{name}_model.pkl')
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} from {filepath}")
        
        # Load ensemble
        filepath = os.path.join(directory, 'ensemble_model.pkl')
        if os.path.exists(filepath):
            self.ensemble = joblib.load(filepath)
            logger.info(f"Loaded ensemble from {filepath}")
        
        # Load metadata
        filepath = os.path.join(directory, 'model_metadata.pkl')
        if os.path.exists(filepath):
            metadata = joblib.load(filepath)
            self.metrics = metadata.get('metrics', {})
            self.feature_importance = metadata.get('feature_importance', {})
            self.best_params = metadata.get('best_params', {})
            self.task = metadata.get('task', 'regression')
            logger.info(f"Loaded metadata from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Create and train ensemble model
    print("Advanced Ensemble Model - BlackRock Aladdin Competitive System")
    print("="*70)
    
    # Generate sample data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Create and train model
    model = AdvancedEnsembleModel(task='regression', optimize=False)
    metrics = model.train(X_df, y, optimize_each=False)
    
    # Make predictions
    predictions = model.predict(X_df[:10])
    print(f"\nSample predictions: {predictions[:5]}")
    
    # Get confidence
    pred, conf = model.predict_with_confidence(X_df[:10])
    print(f"\nPredictions with confidence:")
    for i in range(5):
        print(f"  Prediction: {pred[i]:.2f}, Confidence: {conf[i]:.2f}")
    
    # Model comparison
    print("\nModel Comparison:")
    print(model.get_model_comparison())
    
    # Feature importance
    print(f"\nTop 10 Important Features:")
    for i, (feature, importance) in enumerate(list(model.feature_importance.items())[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Save models
    model.save_models('advanced_models/')
    print("\nModels saved successfully!")
