"""
AQI Prediction System - Phase 4: Advanced Model Development
==========================================================

This script implements advanced machine learning models to achieve the 75% R² target.

Current Status: 69.6% R² baseline (Gap: 5.4% to target)
Target: 75% R² for 24h, 48h, and 72h AQI forecasting

Strategy:
1. XGBoost with hyperparameter tuning (+3-5% R²)
2. LightGBM optimization (+2-4% R²)  
3. Neural Networks with LSTM/Attention (+5-8% R²)
4. Model ensembling (+2-3% R²)

Author: Data Science Team
Date: August 11, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional

# Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Neural networks will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Feature Store Integration
try:
    from phase3_feature_store_api import AQIPeshawarFeatureStore
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    print("⚠️  Feature store not available. Using local data.")
    FEATURE_STORE_AVAILABLE = False

# Hyperparameter Optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("⚠️  Scikit-optimize not available. Using grid search.")
    BAYESIAN_OPT_AVAILABLE = False

warnings.filterwarnings('ignore')

class Phase4ModelDevelopment:
    """
    Advanced Model Development for AQI Prediction
    Target: Achieve 75% R² through advanced ML techniques
    """
    
    def __init__(self):
        """Initialize Phase 4 Model Development"""
        print("🚀 PHASE 4: ADVANCED MODEL DEVELOPMENT")
        print("=" * 50)
        print("🎯 Target: 75% R² for AQI forecasting")
        print("📊 Current baseline: 69.6% R²")
        print("📈 Gap to close: 5.4%")
        print()
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importance = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Model configurations
        self.model_configs = {
            'random_forest': {'enabled': True, 'baseline': True},
            'xgboost': {'enabled': True, 'priority': 1},
            'lightgbm': {'enabled': True, 'priority': 2},
            'neural_network': {'enabled': TENSORFLOW_AVAILABLE, 'priority': 3},
            'ensemble': {'enabled': True, 'priority': 4}
        }

    def step1_load_data_from_feature_store(self):
        """Step 1: Load data from Hopsworks feature store"""
        print("📊 STEP 1: Loading Data from Feature Store")
        print("-" * 42)
        
        try:
            if FEATURE_STORE_AVAILABLE:
                print("🔄 Connecting to Hopsworks feature store...")
                
                # Set up environment variables if not already set
                if not os.getenv('HOPSWORKS_API_KEY'):
                    print("⚠️  Setting up Hopsworks credentials...")
                    # You may need to set these manually
                    print("Please ensure HOPSWORKS_API_KEY and HOPSWORKS_PROJECT are set")
                
                # Initialize feature store
                fs = AQIPeshawarFeatureStore()
                
                # Get training dataset with all relevant features
                print("📋 Creating comprehensive training dataset...")
                categories = ['weather', 'pollution', 'temporal', 'lag_features', 'advanced_features']
                
                df = fs.create_training_dataset(
                    categories=categories,
                    start_date='2025-03-15',  # Start from clean data
                    end_date='2025-08-11'     # Up to current date
                )
                
                print(f"✅ Loaded {len(df)} records from feature store")
                print(f"📈 Features available: {len(df.columns)-2}")  # Exclude timestamp and target
                
            else:
                print("📁 Loading from local feature files...")
                df = pd.read_csv("data_repositories/features/final_features.csv")
                print(f"✅ Loaded {len(df)} records from local file")
            
            # Prepare features and target
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any rows with missing target
            initial_len = len(df)
            df = df.dropna(subset=['aqi_numeric'])
            print(f"📊 Records after removing missing targets: {len(df)} (removed {initial_len - len(df)})")
            
            # Feature columns (exclude timestamp and target)
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi_numeric']]
            self.feature_names = feature_cols
            
            # Handle missing values in features
            print("🔧 Handling missing values...")
            missing_before = df[feature_cols].isnull().sum().sum()
            
            # Forward fill then backward fill for time series data
            df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            missing_after = df[feature_cols].isnull().sum().sum()
            print(f"   Missing values: {missing_before} → {missing_after}")
            
            # Final check for any remaining missing values
            if missing_after > 0:
                print("⚠️  Filling remaining missing values with median...")
                df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            
            print(f"✅ Final dataset: {len(df)} records, {len(feature_cols)} features")
            
            return df, feature_cols
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print("💡 Falling back to local data loading...")
            
            # Fallback to local data
            df = pd.read_csv("data_repositories/features/final_features.csv")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.dropna(subset=['aqi_numeric'])
            
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi_numeric']]
            df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            
            self.feature_names = feature_cols
            print(f"✅ Fallback successful: {len(df)} records, {len(feature_cols)} features")
            
            return df, feature_cols

    def step2_prepare_data_for_modeling(self, df: pd.DataFrame, feature_cols: List[str]):
        """Step 2: Prepare data with proper temporal splits for time series"""
        print("\n🔄 STEP 2: Preparing Data for Modeling")
        print("-" * 38)
        
        try:
            # Sort by timestamp to ensure temporal order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Extract features and target
            X = df[feature_cols].values
            y = df['aqi_numeric'].values
            timestamps = df['timestamp'].values
            
            # Temporal split (80/20) - CRITICAL for time series
            split_idx = int(0.8 * len(df))
            
            self.X_train = X[:split_idx]
            self.X_test = X[split_idx:]
            self.y_train = y[:split_idx]
            self.y_test = y[split_idx:]
            
            train_dates = timestamps[:split_idx]
            test_dates = timestamps[split_idx:]
            
            print(f"📊 Training set: {len(self.X_train)} records")
            print(f"   Date range: {train_dates[0]} to {train_dates[-1]}")
            print(f"📊 Test set: {len(self.X_test)} records") 
            print(f"   Date range: {test_dates[0]} to {test_dates[-1]}")
            
            # Feature scaling for neural networks
            print("⚖️  Setting up feature scaling...")
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Basic statistics
            print(f"\n📈 Target Statistics:")
            print(f"   Training AQI range: {self.y_train.min():.1f} - {self.y_train.max():.1f}")
            print(f"   Test AQI range: {self.y_test.min():.1f} - {self.y_test.max():.1f}")
            print(f"   Training mean: {self.y_train.mean():.2f} ± {self.y_train.std():.2f}")
            print(f"   Test mean: {self.y_test.mean():.2f} ± {self.y_test.std():.2f}")
            
            print("✅ Data preparation completed")
            return True
            
        except Exception as e:
            print(f"❌ Error in data preparation: {str(e)}")
            return False

    def step3_baseline_model_validation(self):
        """Step 3: Validate baseline Random Forest model"""
        print("\n📋 STEP 3: Baseline Model Validation")
        print("-" * 35)
        
        try:
            print("🌲 Training baseline Random Forest model...")
            
            # Use the same configuration as Phase 2 for comparison
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1
            )
            
            rf_model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred_train = rf_model.predict(self.X_train)
            y_pred_test = rf_model.predict(self.X_test)
            
            # Metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            print(f"📊 Baseline Random Forest Results:")
            print(f"   Training R²: {train_r2:.3f}")
            print(f"   Test R²: {test_r2:.3f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test RMSE: {test_rmse:.2f}")
            
            # Store baseline results
            self.models['random_forest'] = rf_model
            self.results['random_forest'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'model_type': 'Random Forest'
            }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['random_forest'] = feature_importance
            
            print(f"🔝 Top 5 features:")
            for i, row in feature_importance.head().iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            # Validation check
            expected_r2 = 0.696  # From Phase 2
            if abs(test_r2 - expected_r2) > 0.05:
                print(f"⚠️  Warning: R² differs from Phase 2 baseline ({expected_r2:.3f})")
                print("   This might indicate data or split differences")
            else:
                print("✅ Baseline validation successful - consistent with Phase 2")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in baseline validation: {str(e)}")
            return False

    def save_step_results(self, step_name: str):
        """Save intermediate results after each step"""
        try:
            results_file = f"data_repositories/features/phase4_{step_name}_results.json"
            
            step_results = {
                'timestamp': datetime.now().isoformat(),
                'step': step_name,
                'models_completed': list(self.results.keys()),
                'results': self.results,
                'best_performance': max([r['test_r2'] for r in self.results.values()]) if self.results else 0
            }
            
            with open(results_file, 'w') as f:
                json.dump(step_results, f, indent=4, default=str)
            
            print(f"💾 Results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {str(e)}")

    def run_phase4_step1_to_3(self):
        """Run Phase 4 Steps 1-3: Data loading and baseline validation"""
        print("\n🚀 EXECUTING PHASE 4 - STEPS 1-3")
        print("=" * 40)
        
        # Step 1: Load data
        df, feature_cols = self.step1_load_data_from_feature_store()
        if df is None:
            return False
        
        # Step 2: Prepare data
        if not self.step2_prepare_data_for_modeling(df, feature_cols):
            return False
        
        # Step 3: Baseline validation
        if not self.step3_baseline_model_validation():
            return False
        
        # Save intermediate results
        self.save_step_results("baseline")
        
        print("\n🎉 STEPS 1-3 COMPLETED SUCCESSFULLY!")
        print("=" * 35)
        print(f"✅ Data loaded: {len(self.X_train) + len(self.X_test)} records")
        print(f"✅ Features prepared: {len(self.feature_names)} features")
        print(f"✅ Baseline established: {self.results['random_forest']['test_r2']:.3f} R²")
        print(f"🎯 Target: 0.750 R² (Gap: {0.750 - self.results['random_forest']['test_r2']:.3f})")
        print("\n🚀 Ready for advanced model development!")
        
        return True

def main():
    """Run Phase 4 Model Development - Steps 1-3"""
    # Set up environment
    os.environ.setdefault('HOPSWORKS_API_KEY', 'VjkiieLcPvPdjYXt.JUN053a8LMAgn4orHy6Enc0uS1PRONLUTGIkbHe8zexoRFeyxjmyflMzTtQ1dU4l')
    os.environ.setdefault('HOPSWORKS_PROJECT', 'aqi_prediction_pekhawar')
    
    # Initialize and run
    phase4 = Phase4ModelDevelopment()
    success = phase4.run_phase4_step1_to_3()
    
    if success:
        print("\n🎯 PHASE 4 FOUNDATION READY!")
        print("Next: Implement XGBoost, LightGBM, and Neural Networks")
    else:
        print("\n⚠️  PHASE 4 SETUP INCOMPLETE")
        print("Please review errors above")

if __name__ == "__main__":
    main()
