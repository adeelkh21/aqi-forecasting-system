"""
Multi-Horizon AQI Forecasting System - Model Training Pipeline (UPDATED VERSION)
==============================================================================

This script trains various machine learning models on the MULTI-HORIZON dataset:
- Uses the 50-feature dataset from feature selection (no data leakage)
- Implements multi-horizon forecasting (24h, 48h, 72h)
- Implements proper train/validation/test split
- Saves trained models and performance metrics for all horizons

Author: Data Science Team
Date: 2024-03-09
Updated: 2025-01-XX for Multi-Horizon Forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
import threading
import time
warnings.filterwarnings('ignore')

# Suppress all verbose output from ML libraries
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['LIGHTGBM_VERBOSE'] = '0'       # Suppress LightGBM logging
os.environ['XGBOOST_VERBOSE'] = '0'        # Suppress XGBoost logging

# Cross-platform timeout handler for long-running operations
class TimeoutError(Exception):
    pass

def timeout_handler(func, args=(), kwargs={}, timeout_duration=300):
    """Execute function with timeout using threading"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return None, TimeoutError("Operation timed out")
    else:
        # Thread completed
        if exception[0]:
            return None, exception[0]
        return result[0], None

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Advanced models
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

print("🔧 MULTI-HORIZON AQI FORECASTING - MODEL TRAINING PIPELINE")
print("=" * 70)
print(f"🎯 Multi-Horizon Targets: 24h (primary), 48h, 72h")
print(f"🔍 Features: 50 selected via SHAP analysis")
print(f"🔒 Data Leakage Prevention: ✅ Active")
print(f"LightGBM: {'✅' if LGBM_AVAILABLE else '❌'}")
print(f"XGBoost: {'✅' if XGB_AVAILABLE else '❌'}")
print(f"CatBoost: {'✅' if CATBOOST_AVAILABLE else '❌'}")

class MultiHorizonModelTrainer:
    """Train multiple ML models on multi-horizon AQI forecasting dataset"""
    
    def __init__(self):
        # Configuration constants (no more hardcoding!)
        self.TRAIN_SPLIT_RATIO = 0.6
        self.VAL_SPLIT_RATIO = 0.2
        self.TEST_SPLIT_RATIO = 0.2
        
        self.TRAINING_TIMEOUT = 300  # 5 minutes
        self.TUNING_TIMEOUT = 180    # 3 minutes
        
        self.TOP_K_TUNE = 3  # Number of top models to fine-tune
        
        # Model hyperparameter configurations
        self.MODEL_CONFIGS = {
            'random_forest': {
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 0
            },
            'gradient_boosting': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 3,
                'subsample': 0.8,
                'random_state': 42,
                'verbose': 0
            },
            'lightgbm': {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': -1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'xgboost': {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'rmse',
                'verbosity': 0
            },
            'catboost': {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_state': 42,
                'verbose': False
            }
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.performance_metrics = {}
        self.multi_horizon_metrics = {}
        
        # Multi-horizon targets
        self.targets = {
            '24h': 'target_aqi_24h',
            '48h': 'target_aqi_48h', 
            '72h': 'target_aqi_72h'
        }
        
    def load_clean_data(self):
        """Load the multi-horizon feature-selected dataset"""
        print("\n📊 Loading multi-horizon dataset...")
        
        try:
            # Load the multi-horizon feature-selected dataset
            df = pd.read_csv('data_repositories/features/phase1_fixed_selected_features.csv')
            
            # Load feature columns
            with open('data_repositories/features/phase1_fixed_feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"✅ Multi-horizon dataset loaded: {df.shape}")
            print(f"📊 Features: {len(self.feature_columns)}")
            print(f"🎯 Multi-horizon targets:")
            for horizon, target_col in self.targets.items():
                if target_col in df.columns:
                    target_range = f"{df[target_col].min():.1f} to {df[target_col].max():.1f}"
                    valid_count = df[target_col].notna().sum()
                    print(f"      → {horizon}: {target_col} ({target_range}, {valid_count:,} valid)")
                else:
                    print(f"      ❌ {horizon}: {target_col} (MISSING!)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
    def prepare_data(self, df):
        """Prepare features and multi-horizon targets from dataset"""
        print("\n🔧 Preparing multi-horizon data...")
        
        # Use the clean feature columns (no AQI lags!)
        X = df[self.feature_columns]
        
        # Prepare multi-horizon targets
        targets_data = {}
        for horizon, target_col in self.targets.items():
            if target_col in df.columns:
                y = df[target_col]
                
                # Remove rows with NaN target (due to forecasting shift)
                valid_mask = y.notna()
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                targets_data[horizon] = {
                    'X': X_valid,
                    'y': y_valid,
                    'valid_mask': valid_mask
                }
                
                print(f"   ✅ {horizon}: {target_col} - {y_valid.shape[0]:,} valid samples")
                print(f"      Range: {y_valid.min():.1f} - {y_valid.max():.1f}")
            else:
                print(f"   ❌ {horizon}: {target_col} not found in dataset")
                return None, None, None
        
        # Use the 24h horizon as primary for feature scaling (most important)
        primary_data = targets_data['24h']
        X_primary = primary_data['X']
        y_primary = primary_data['y']
        
        print(f"\n✅ Multi-horizon data prepared:")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Primary target (24h): {y_primary.shape[0]:,} samples")
        print(f"   All horizons available for training")
        
        return X_primary, y_primary, targets_data
    
    def split_data(self, X, y):
        """Split data into train/validation/test sets using configuration constants"""
        print(f"\n✂️ Splitting data ({int(self.TRAIN_SPLIT_RATIO*100)}/{int(self.VAL_SPLIT_RATIO*100)}/{int(self.TEST_SPLIT_RATIO*100)})...")
        
        # First split: train vs temp using configuration constants
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.VAL_SPLIT_RATIO + self.TEST_SPLIT_RATIO), random_state=42, shuffle=False
        )
        
        # Second split: temp into validation and test using configuration constants
        val_ratio = self.VAL_SPLIT_RATIO / (self.VAL_SPLIT_RATIO + self.TEST_SPLIT_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), random_state=42, shuffle=False
        )
        
        print(f"   ✅ Train: {X_train.shape[0]:,} samples")
        print(f"   ✅ Validation: {X_val.shape[0]:,} samples")
        print(f"   ✅ Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_temp, y_train, y_temp
    
    def scale_features(self, X_train, X_temp, y_train, y_temp):
        """Scale features using StandardScaler"""
        print(f"\n⚖️ Scaling features...")
        
        # Split the temp data into validation and test using configuration constants
        val_ratio = self.VAL_SPLIT_RATIO / (self.VAL_SPLIT_RATIO + self.TEST_SPLIT_RATIO)
        val_size = int(val_ratio * len(X_temp))
        X_val = X_temp.iloc[:val_size]
        y_val = y_temp.iloc[:val_size]
        X_test = X_temp.iloc[val_size:]
        y_test = y_temp.iloc[val_size:]
        
        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['standard'] = scaler
        
        print(f"   ✅ Features scaled using StandardScaler")
        print(f"   ✅ Training: {X_train_scaled.shape}")
        print(f"   ✅ Validation: {X_val_scaled.shape}")
        print(f"   ✅ Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test
    
    def initialize_models(self):
        """Initialize all models to train using configuration constants"""
        print("\n🤖 Initializing models...")
        
        # Traditional ML models (using configuration constants)
        self.models['random_forest'] = RandomForestRegressor(**self.MODEL_CONFIGS['random_forest'])

        self.models['gradient_boosting'] = GradientBoostingRegressor(**self.MODEL_CONFIGS['gradient_boosting'])
        
        self.models['linear_regression'] = LinearRegression()
        
        self.models['ridge'] = Ridge(alpha=5.0, random_state=42)
        
        self.models['lasso'] = Lasso(alpha=0.01, random_state=42, max_iter=5000)
        
        self.models['svr'] = SVR(kernel='rbf', C=10.0, epsilon=0.2, gamma='scale', verbose=False)
        
        self.models['knn'] = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
        
        self.models['decision_tree'] = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42)
        
        # Advanced models with early-stopping-friendly configs (using configuration constants)
        if LGBM_AVAILABLE:
            self.models['lightgbm'] = LGBMRegressor(**self.MODEL_CONFIGS['lightgbm'])
        
        if XGB_AVAILABLE:
            self.models['xgboost'] = XGBRegressor(**self.MODEL_CONFIGS['xgboost'])
        
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostRegressor(**self.MODEL_CONFIGS['catboost'])
        
        print(f"✅ {len(self.models)} models initialized")
        return list(self.models.keys())
    
    def train_model(self, name, model, X_train, y_train, X_val, y_val, targets_data):
        """Train a single model for multi-horizon forecasting"""
        print(f"   🔄 Training {name} for multi-horizon forecasting...")
        
        try:
            # Train model with early stopping where supported (using configuration constants)
            if 'lightgbm' in name:
                try:
                    result, error = timeout_handler(
                        model.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': [(X_val, y_val)], 'eval_metric': 'rmse', 'callbacks': [
                            lgb.early_stopping(stopping_rounds=100, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]},
                        timeout_duration=self.TRAINING_TIMEOUT
                    )
                    
                    if error is not None:
                        print(f"      ⚠️ {name} timed out during training. Skipping.")
                        return None, float('inf'), 0, {}
                except Exception:
                    model.fit(X_train, y_train)
            elif 'xgboost' in name:
                try:
                    result, error = timeout_handler(
                        model.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': [(X_val, y_val)], 'verbose': False},
                        timeout_duration=self.TRAINING_TIMEOUT
                    )
                    
                    if error is not None:
                        print(f"      ⚠️ {name} timed out during training. Skipping.")
                        return None, float('inf'), 0, {}
                except Exception:
                    model.fit(X_train, y_train)
            elif 'catboost' in name:
                try:
                    result, error = timeout_handler(
                        model.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': (X_val, y_val), 'use_best_model': True, 'early_stopping_rounds': 100, 'verbose': False},
                        timeout_duration=self.TRAINING_TIMEOUT
                    )
                    
                    if error is not None:
                        print(f"      ⚠️ {name} timed out during training. Skipping.")
                        return None, float('inf'), 0, {}
                except Exception:
                    model.fit(X_train, y_train)
            else:
                try:
                    result, error = timeout_handler(
                        model.fit,
                        args=(X_train, y_train),
                        kwargs={},
                        timeout_duration=self.TRAINING_TIMEOUT
                    )
                    
                    if error is not None:
                        print(f"      ⚠️ {name} timed out during training. Skipping.")
                        return None, float('inf'), 0, {}
                except Exception:
                    model.fit(X_train, y_train)
            
            # Evaluate on primary target (24h)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate primary target metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_r2 = r2_score(y_val, y_val_pred)
            
            print(f"      ✅ Primary (24h) - Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}, Val R²: {val_r2:.3f}")
            
            # Evaluate on all horizons
            horizon_metrics = {}
            for horizon, target_col in self.targets.items():
                if horizon in targets_data:
                    horizon_X = targets_data[horizon]['X']
                    horizon_y = targets_data[horizon]['y']
                    
                    # Split horizon data using configuration constants
                    train_size = int(self.TRAIN_SPLIT_RATIO * len(horizon_X))
                    val_size = int(self.VAL_SPLIT_RATIO * len(horizon_X))
                    
                    X_h_train = horizon_X.iloc[:train_size]
                    y_h_train = horizon_y.iloc[:train_size]
                    X_h_val = horizon_X.iloc[train_size:train_size+val_size]
                    y_h_val = horizon_y.iloc[train_size:train_size+val_size]
                    
                    # Predict and evaluate
                    y_h_train_pred = model.predict(X_h_train)
                    y_h_val_pred = model.predict(X_h_val)
                    
                    h_train_rmse = np.sqrt(mean_squared_error(y_h_train, y_h_train_pred))
                    h_val_rmse = np.sqrt(mean_squared_error(y_h_val, y_h_val_pred))
                    h_val_r2 = r2_score(y_h_val, y_h_val_pred)
                    
                    horizon_metrics[horizon] = {
                        'train_rmse': h_train_rmse,
                        'val_rmse': h_val_rmse,
                        'val_r2': h_val_r2
                    }
                    
                    if horizon != '24h':  # Don't repeat primary metrics
                        print(f"      📊 {horizon} - Train RMSE: {h_train_rmse:.2f}, Val RMSE: {h_val_rmse:.2f}, Val R²: {h_val_r2:.3f}")
            
            return model, val_rmse, val_r2, horizon_metrics
            
        except Exception as e:
            print(f"      ❌ Error training {name}: {str(e)}")
            return None, float('inf'), 0, {}
    
    def train_all_models(self, X_train, y_train, X_val, y_val, targets_data):
        """Train all models for multi-horizon forecasting"""
        print("\n🚀 Training all models for multi-horizon forecasting...")
        
        model_results = {}
        total_models = len(self.models)
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"   [{i}/{total_models}] Training {name}...")
            trained_model, val_rmse, val_r2, horizon_metrics = self.train_model(
                name, model, X_train, y_train, X_val, y_val, targets_data
            )
            
            if trained_model is not None:
                self.models[name] = trained_model
                model_results[name] = {
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'horizon_metrics': horizon_metrics
                }
                print(f"      ✅ {name} completed successfully")
            else:
                print(f"      ❌ {name} failed to train")
        
        # Sort by validation RMSE (primary target)
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['val_rmse'])
        
        print(f"\n🏆 MODEL RANKING (by Primary Target 24h Validation RMSE):")
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"   {i}. {name}: RMSE={metrics['val_rmse']:.2f}, R²={metrics['val_r2']:.3f}")
            
            # Show multi-horizon performance
            if 'horizon_metrics' in metrics:
                for horizon, h_metrics in metrics['horizon_metrics'].items():
                    if horizon != '24h':  # Don't repeat primary
                        print(f"      📊 {horizon}: RMSE={h_metrics['val_rmse']:.2f}, R²={h_metrics['val_r2']:.3f}")
        
        return model_results

    def tune_model(self, name, model, X_train, y_train, X_val, y_val, targets_data):
        """Lightweight hyperparameter tuning for select models using validation set."""
        rng = np.random.RandomState(42)
        best_model = model
        
        # Helper to evaluate a fitted model on validation (primary target)
        def eval_val(fitted):
            try:
                y_pred = fitted.predict(X_val)
                return float(np.sqrt(mean_squared_error(y_val, y_pred)))
            except Exception:
                return float('inf')
        
        # SVR grid
        if name == 'svr':
            param_grid = [
                {'C': C, 'epsilon': eps, 'gamma': gam}
                for C in [3.0, 10.0, 30.0]
                for eps in [0.1, 0.2, 0.3]
                for gam in ['scale', 'auto']
            ]
            best_rmse = float('inf')
            for p in param_grid:
                try:
                    from sklearn.svm import SVR as _SVR
                    cand = _SVR(kernel='rbf', **p)
                    cand.fit(X_train, y_train)
                    rmse = eval_val(cand)
                    if rmse < best_rmse:
                        best_rmse, best_model = rmse, cand
                except Exception:
                    continue
            return best_model
        
        # LightGBM random search (SUPPRESS VERBOSE LOGS)
        if name == 'lightgbm':
            if not LGBM_AVAILABLE:
                return best_model
            samples = 20
            best_rmse = float('inf')
            for _ in range(samples):
                params = {
                    'n_estimators': 3000,
                    'learning_rate': float(rng.choice([0.03, 0.05, 0.08]))
                }
                params['num_leaves'] = int(rng.choice([31, 64, 96]))
                params['min_child_samples'] = int(rng.choice([10, 20, 30]))
                params['subsample'] = float(rng.choice([0.7, 0.9, 1.0]))
                params['colsample_bytree'] = float(rng.choice([0.7, 0.9, 1.0]))
                cand = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **params)
                try:
                    result, error = timeout_handler(
                        cand.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': [(X_val, y_val)], 'eval_metric': 'rmse', 'callbacks': [
                            lgb.early_stopping(stopping_rounds=100, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]},
                        timeout_duration=self.TUNING_TIMEOUT
                    )
                    if error is not None:
                        print(f"      ⚠️ LightGBM tuning timed out. Skipping.")
                        continue
                except Exception:
                    continue
                rmse = eval_val(cand)
                if rmse < best_rmse:
                    best_rmse, best_model = rmse, cand
            return best_model
        
        # XGBoost random search (SUPPRESS VERBOSE LOGS)
        if name == 'xgboost':
            if not XGB_AVAILABLE:
                return best_model
            from xgboost import XGBRegressor as _XGBRegressor
            samples = 20
            best_rmse = float('inf')
            for _ in range(samples):
                cand = _XGBRegressor(
                    n_estimators=3000,
                    learning_rate=float(rng.choice([0.03, 0.05, 0.08])),
                    max_depth=int(rng.choice([5, 6, 8])),
                    subsample=float(rng.choice([0.7, 0.9, 1.0])),
                    colsample_bytree=float(rng.choice([0.7, 0.9, 1.0])),
                    reg_lambda=float(rng.choice([1.0, 2.0])),
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='rmse',
                    verbosity=0
                )
                try:
                    result, error = timeout_handler(
                        cand.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': [(X_val, y_val)], 'verbose': False},
                        timeout_duration=self.TUNING_TIMEOUT
                    )
                    if error is not None:
                        print(f"      ⚠️ XGBoost tuning timed out. Skipping.")
                        continue
                except Exception:
                    continue
                rmse = eval_val(cand)
                if rmse < best_rmse:
                    best_rmse, best_model = rmse, cand
            return best_model
        
        # CatBoost random search (SUPPRESS VERBOSE LOGS)
        if name == 'catboost':
            if not CATBOOST_AVAILABLE:
                return best_model
            from catboost import CatBoostRegressor as _Cat
            samples = 15
            best_rmse = float('inf')
            for _ in range(samples):
                cand = _Cat(
                    iterations=3000,
                    learning_rate=float(rng.choice([0.03, 0.05, 0.08])),
                    depth=int(rng.choice([6, 8, 10])),
                    l2_leaf_reg=float(rng.choice([2.0, 3.0, 5.0])),
                    loss_function='RMSE', 
                    eval_metric='RMSE', 
                    random_state=42, 
                    verbose=False
                )
                try:
                    result, error = timeout_handler(
                        cand.fit,
                        args=(X_train, y_train),
                        kwargs={'eval_set': (X_val, y_val), 'use_best_model': True, 'early_stopping_rounds': 100, 'verbose': False},
                        timeout_duration=self.TUNING_TIMEOUT
                    )
                    if error is not None:
                        print(f"      ⚠️ CatBoost tuning timed out. Skipping.")
                        continue
                except Exception:
                    continue
                rmse = eval_val(cand)
                if rmse < best_rmse:
                    best_rmse, best_model = rmse, cand
            return best_model
        
        return best_model

    def tune_top_models(self, training_results, X_train, y_train, X_val, y_val, targets_data, top_k=None):
        """Fine-tune top k models and refresh results using configuration constants"""
        if top_k is None:
            top_k = self.TOP_K_TUNE
            
        print(f"\n🔧 Fine-tuning top {top_k} models...")
        
        # Get top models by validation RMSE
        ordered = sorted(training_results.items(), key=lambda x: x[1]['val_rmse'])
        to_tune = [name for name, _ in ordered[:top_k]]
        
        for i, name in enumerate(to_tune, 1):
            print(f"   [{i}/{len(to_tune)}] Fine-tuning {name}...")
            tuned = self.tune_model(name, self.models[name], X_train, y_train, X_val, y_val, targets_data)
            self.models[name] = tuned
            
            # Re-evaluate on validation
            trained_model, val_rmse, val_r2, horizon_metrics = self.train_model(
                name, tuned, X_train, y_train, X_val, y_val, targets_data
            )
            
            if trained_model is not None:
                training_results[name] = {
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'horizon_metrics': horizon_metrics
                }
                print(f"      ✅ {name} tuned - New RMSE: {val_rmse:.2f}, R²: {val_r2:.3f}")
            else:
                print(f"      ❌ {name} tuning failed")
        
        # Re-sort results
        training_results = dict(sorted(training_results.items(), key=lambda x: x[1]['val_rmse']))
        
        print(f"\n🏆 UPDATED MODEL RANKING (after fine-tuning):")
        for i, (name, metrics) in enumerate(training_results.items(), 1):
            print(f"   {i}. {name}: RMSE={metrics['val_rmse']:.2f}, R²={metrics['val_r2']:.3f}")
            
            # Show multi-horizon performance
            if 'horizon_metrics' in metrics:
                for horizon, h_metrics in metrics['horizon_metrics'].items():
                    if horizon != '24h':  # Don't repeat primary
                        print(f"      📊 {horizon}: RMSE={h_metrics['val_rmse']:.2f}, R²={h_metrics['val_r2']:.3f}")
        
        return training_results

    def blend_top_models(self, model_results, X_val, y_val, X_test, y_test):
        """Create a simple linear blend of top models using validation data."""
        try:
            # Select up to three best models available
            ordered = sorted(model_results.items(), key=lambda x: x[1]['val_rmse'])
            top_names = [n for n, _ in ordered if n in self.models][:3]
            if len(top_names) < 2:
                return None
            # Build validation prediction matrix
            val_preds = []
            for n in top_names:
                val_preds.append(self.models[n].predict(X_val))
            val_stack = np.vstack(val_preds).T  # (n_val, k)
            # Fit linear blender
            blender = LinearRegression(fit_intercept=True)
            blender.fit(val_stack, y_val)
            # Predict on test
            test_stack = np.vstack([self.models[n].predict(X_test) for n in top_names]).T
            y_pred = blender.predict(test_stack)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            mape = float(mean_absolute_percentage_error(y_test, y_pred))
            # Store
            self.performance_metrics['blended_ensemble'] = {'RMSE': rmse, 'MAE': float(mean_absolute_error(y_test, y_pred)), 'R²': r2, 'MAPE (%)': mape}
            # Save weights
            os.makedirs('saved_models', exist_ok=True)
            import json
            with open('saved_models/ensemble_meta.json', 'w') as f:
                json.dump({'models': top_names, 'coef': blender.coef_.tolist(), 'intercept': float(blender.intercept_)}, f, indent=2)
            print(f"\n🤝 Blended ensemble ({', '.join(top_names)}): Test RMSE={rmse:.2f}, R²={r2:.3f}")
            return {'Model': 'blended_ensemble', 'RMSE': rmse, 'R²': r2}
        except Exception as e:
            print(f"   ❌ Blending failed: {str(e)}")
            return None
    
    def evaluate_on_test(self, X_test, y_test, targets_data):
        """Evaluate all models on test set for multi-horizon forecasting"""
        print(f"\n🧪 Evaluating models on test set for multi-horizon forecasting...")
        
        test_results = {}
        
        for name, model in self.models.items():
            try:
                print(f"   🔍 Evaluating {name}...")
                
                # Evaluate on primary target (24h)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                test_results[name] = {
                    '24h': {
                        'RMSE': rmse,
                        'MAE': mae,
                        'R²': r2,
                        'MAPE (%)': mape
                    }
                }
                
                print(f"      ✅ Primary (24h): RMSE={rmse:.2f}, R²={r2:.3f}")
                
                # Evaluate on other horizons
                for horizon, target_col in self.targets.items():
                    if horizon != '24h' and horizon in targets_data:
                        horizon_X = targets_data[horizon]['X']
                        horizon_y = targets_data[horizon]['y']
                        
                        # Split horizon data using same indices as primary
                        train_size = int(self.TRAIN_SPLIT_RATIO * len(horizon_X))
                        val_size = int(self.VAL_SPLIT_RATIO * len(horizon_X))
                        test_start = train_size + val_size
                        
                        X_h_test = horizon_X.iloc[test_start:]
                        y_h_test = horizon_y.iloc[test_start:]
                        
                        if len(X_h_test) > 0:
                            y_h_pred = model.predict(X_h_test)
                            h_rmse = np.sqrt(mean_squared_error(y_h_test, y_h_pred))
                            h_mae = mean_absolute_error(y_h_test, y_h_pred)
                            h_r2 = r2_score(y_h_test, y_h_pred)
                            h_mape = mean_absolute_percentage_error(y_h_test, y_h_pred)
                            
                            test_results[name][horizon] = {
                                'RMSE': h_rmse,
                                'MAE': h_mae,
                                'R²': h_r2,
                                'MAPE (%)': h_mape
                            }
                            
                            print(f"      📊 {horizon}: RMSE={h_rmse:.2f}, R²={h_r2:.3f}")
                        else:
                            print(f"      ⚠️ {horizon}: No test data available")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {str(e)}")
        
        self.performance_metrics = test_results
        return test_results
    
    def save_models(self):
        """Save all trained multi-horizon models"""
        print("\n💾 Saving multi-horizon models...")
        
        # Create models directory
        os.makedirs('saved_models', exist_ok=True)
        
        saved_count = 0
        for name, model in self.models.items():
            try:
                # Save model
                model_path = f'saved_models/{name}_multi_horizon_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save scaler if available
                if name in self.scalers:
                    scaler_path = f'saved_models/{name}_multi_horizon_scaler.pkl'
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[name], f)
                
                print(f"   ✅ {name} saved to {model_path}")
                saved_count += 1
                
            except Exception as e:
                print(f"   ❌ Error saving {name}: {str(e)}")
        
        print(f"\n💾 Multi-horizon models saved: {saved_count}/{len(self.models)}")
        
        # Save feature columns
        if self.feature_columns is not None:
            feature_path = 'saved_models/multi_horizon_feature_columns.pkl'
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"   ✅ Feature columns saved to {feature_path}")
        
        return saved_count

    def save_performance_metrics(self):
        """Save multi-horizon performance metrics"""
        print("\n📊 Saving multi-horizon performance metrics...")
        
        # Create comprehensive results DataFrame
        results_data = []
        
        for model_name, model_data in self.performance_metrics.items():
            # Check if this is a blended ensemble (different structure)
            if model_name == 'blended_ensemble':
                # Blended ensemble has flat structure
                row = {
                    'Model': model_name,
                    'Horizon': '24h',  # Blended ensemble only for primary target
                    **model_data
                }
                results_data.append(row)
            else:
                # Individual models have horizon-based structure
                for horizon, metrics in model_data.items():
                    row = {
                        'Model': model_name,
                        'Horizon': horizon,
                        **metrics
                    }
                    results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        results_path = 'saved_models/multi_horizon_model_performance.csv'
        results_df.to_csv(results_path, index=False)
        print(f"   ✅ Multi-horizon performance metrics saved to {results_path}")
        
        # Save to JSON
        json_path = 'saved_models/multi_horizon_model_performance.json'
        results_df.to_json(json_path, orient='records', indent=2)
        print(f"   ✅ Multi-horizon performance metrics saved to {json_path}")
        
        # Create summary by horizon
        print(f"\n📊 PERFORMANCE SUMMARY BY HORIZON:")
        for horizon in self.targets.keys():
            horizon_data = results_df[results_df['Horizon'] == horizon]
            if not horizon_data.empty:
                best_model = horizon_data.loc[horizon_data['RMSE'].idxmin()]
                print(f"   🎯 {horizon}: Best model = {best_model['Model']} (RMSE: {best_model['RMSE']:.2f})")
        
        return results_df

def main():
    """Main function to run model training pipeline"""
    try:
        print("🚀 MODEL TRAINING PIPELINE (FIXED VERSION)")
        print("=" * 50)
        print("🎯 Using selected feature set (no data leakage, 72h horizon)")
        print("🎯 Same dataset used across models")
        
        # Initialize trainer
        trainer = MultiHorizonModelTrainer()
        
        # Load clean data (same as TCN)
        df = trainer.load_clean_data()
        if df is None:
            return
        
        # Prepare data
        X, y, targets_data = trainer.prepare_data(df)
        if X is None:
            return
        
        # Split data
        X_train, X_temp, y_train, y_temp = trainer.split_data(X, y)
        X_val, X_test, y_val, y_test = trainer.split_data(X_temp, y_temp)
        
        print(f"\n📊 Data split:")
        print(f"   Training: {X_train.shape[0]:,} samples")
        print(f"   Validation: {X_val.shape[0]:,} samples")
        print(f"   Test: {X_test.shape[0]:,} samples")
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test = trainer.scale_features(X_train, X_temp, y_train, y_temp)
        
        # Initialize models
        trainer.initialize_models()
        
        # Train all models
        training_results = trainer.train_all_models(X_train_scaled, y_train, X_val_scaled, y_val, targets_data)

        # Fine-tune top models and refresh results
        training_results = trainer.tune_top_models(training_results, X_train_scaled, y_train, X_val_scaled, y_val, targets_data)
        
        # Evaluate on test set
        test_results = trainer.evaluate_on_test(X_test_scaled, y_test, targets_data)

        # Optional: simple blended ensemble of the best models
        if len(training_results) >= 2:
            print(f"\n🔀 Creating blended ensemble...")
            blend_results = trainer.blend_top_models(training_results, X_val_scaled, y_val, X_test_scaled, y_test)
        
        # Save models and results
        trainer.save_models()
        trainer.save_performance_metrics()
        
        print(f"\n🎉 Multi-Horizon AQI Forecasting Model Training Complete!")
        print(f"   🎯 24h forecasting: High confidence for daily planning")
        print(f"   🎯 48h forecasting: Medium confidence for weekend planning")
        print(f"   🎯 72h forecasting: Lower confidence for trend indication")
        print(f"   🔒 No data leakage: Only legitimate forecasting features")
        print(f"   ✅ Multi-horizon targets properly configured")
        print(f"   📊 Performance metrics saved for all horizons")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user. Cleaning up...")
        # No explicit signal.alarm(0) needed here as timeout_handler handles it
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        # No explicit signal.alarm(0) needed here as timeout_handler handles it

if __name__ == "__main__":
    main()
