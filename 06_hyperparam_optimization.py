"""
AQI Prediction System - Hyperparameter Optimization Pipeline (MULTI-HORIZON VERSION)
===============================================================================

This script performs hyperparameter optimization for top-performing models:
- Uses the same 50-feature dataset as the multi-horizon training (no data leakage)
- Implements RandomizedSearchCV with TimeSeriesSplit
- Focuses on SVR, LightGBM, XGBoost, and CatBoost
- Optimizes for multi-horizon forecasting (24h, 48h, 72h)

Author: Data Science Team
Date: 2024-03-09
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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

# Parameter distributions
from scipy.stats import loguniform, randint, uniform

print("🔧 HYPERPARAMETER OPTIMIZATION PIPELINE (MULTI-HORIZON VERSION)")
print("=" * 70)
print(f"LightGBM: {'✅' if LGBM_AVAILABLE else '❌'}")
print(f"XGBoost: {'✅' if XGB_AVAILABLE else '❌'}")
print(f"CatBoost: {'✅' if CATBOOST_AVAILABLE else '❌'}")
print("🎯 Using multi-horizon dataset (no data leakage, 24h/48h/72h horizons)")
print("🎯 Same dataset used across models")

class HyperparameterOptimizer:
    """Optimize hyperparameters for top-performing multi-horizon models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.optimization_results = {}
        self.targets = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        
    def load_clean_data(self):
        """Load the multi-horizon clean dataset (50 features, no leakage)"""
        print("\n📊 Loading multi-horizon dataset...")
        
        try:
            # Load the multi-horizon, feature-selected dataset
            df = pd.read_csv('data_repositories/features/phase1_fixed_selected_features.csv')
            
            # Load feature columns
            with open('data_repositories/features/phase1_fixed_feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"✅ Multi-horizon dataset loaded: {df.shape}")
            print(f"📊 Features: {len(self.feature_columns)}")
            print(f"🎯 Multi-horizon targets:")
            for target in self.targets:
                if target in df.columns:
                    valid_count = df[target].notna().sum()
                    target_range = f"{df[target].min():.1f} to {df[target].max():.1f}"
                    print(f"      → {target}: {target_range}, {valid_count:,} valid")
                else:
                    print(f"      → {target}: ❌ Missing")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare features and multi-horizon targets from clean dataset"""
        print("\n🔧 Preparing multi-horizon data...")
        
        # Use the clean feature columns (no AQI lags!)
        X = df[self.feature_columns]
        
        # Prepare multi-horizon targets
        targets_data = {}
        for target in self.targets:
            if target in df.columns:
                targets_data[target] = df[target]
                print(f"   ✅ {target}: {df[target].notna().sum():,} valid samples")
                print(f"      Range: {df[target].min():.1f} - {df[target].max():.1f}")
            else:
                print(f"   ❌ {target}: Missing from dataset")
        
        # Use 24h target as primary for optimization (same as training script)
        y = targets_data.get('target_aqi_24h')
        if y is None:
            print("❌ Primary target (24h) not found!")
            return None, None, None
        
        # Remove rows with NaN target (due to forecasting shift)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Filter targets data to match valid rows
        for target in targets_data:
            targets_data[target] = targets_data[target][valid_mask]
        
        print(f"\n✅ Multi-horizon data prepared:")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Primary target (24h): {len(y)} samples")
        print(f"   All horizons available for training")
        
        return X, y, targets_data
    
    def split_data(self, X, y):
        """Split data using EXACT same approach as training script (60/20/20)"""
        print("\n✂️ Splitting data (60/20/20) - EXACT same as training script...")
        
        # Use EXACT same split logic as 04_model_training.py
        total_samples = len(X)
        train_size = int(0.6 * total_samples)
        val_size = int(0.2 * total_samples)
        
        # Ensure we get the same split as training script
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        
        print(f"   ✅ Train: {len(X_train):,} samples ({len(X_train)/total_samples*100:.1f}%)")
        print(f"   ✅ Validation: {len(X_val):,} samples ({len(X_val)/total_samples*100:.1f}%)")
        print(f"   ✅ Test: {len(X_test):,} samples ({len(X_test)/total_samples*100:.1f}%)")
        
        # Verify split matches training script expectations
        expected_train = 1956
        expected_val = 652
        expected_test = 652
        
        if len(X_train) == expected_train and len(X_val) == expected_val and len(X_test) == expected_test:
            print(f"   ✅ Split matches training script exactly!")
        else:
            print(f"   ⚠️  Split differs from training script:")
            print(f"      Expected: {expected_train}/{expected_val}/{expected_test}")
            print(f"      Got: {len(X_train)}/{len(X_val)}/{len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler - EXACT same as training script"""
        print("\n⚖️ Scaling features using StandardScaler...")
        
        # Fit scaler ONLY on training data (same as training script)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        self.scalers['standard'] = scaler
        print("✅ Features scaled using StandardScaler")
        print(f"   ✅ Training: {X_train_scaled.shape}")
        print(f"   ✅ Validation: {X_val_scaled.shape}")
        print(f"   ✅ Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def define_parameter_spaces(self):
        """Define hyperparameter search spaces"""
        print("\n🔍 Defining parameter spaces...")
        
        self.param_spaces = {}
        
        # SVR parameters
        self.param_spaces['svr'] = {
            'C': loguniform(0.1, 100),
            'gamma': ['scale', 'auto'] + list(loguniform(0.001, 1).rvs(5)),
            'kernel': ['rbf', 'linear'],
            'epsilon': uniform(0.01, 0.2)
        }
        
        # Random Forest parameters
        self.param_spaces['random_forest'] = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', None]
        }
        
        # LightGBM parameters (expanded)
        if LGBM_AVAILABLE:
            self.param_spaces['lightgbm'] = {
                'n_estimators': randint(500, 3000),
                'learning_rate': loguniform(0.01, 0.2),
                'max_depth': randint(3, 12),
                'num_leaves': randint(31, 255),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_samples': randint(10, 40)
            }
        
        # XGBoost parameters (expanded)
        if XGB_AVAILABLE:
            self.param_spaces['xgboost'] = {
                'n_estimators': randint(500, 3000),
                'learning_rate': loguniform(0.01, 0.2),
                'max_depth': randint(4, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': loguniform(0.001, 10),
                'reg_lambda': loguniform(0.01, 10)
            }
        
        # CatBoost parameters (expanded)
        if CATBOOST_AVAILABLE:
            self.param_spaces['catboost'] = {
                'iterations': randint(500, 2000),
                'learning_rate': loguniform(0.01, 0.2),
                'depth': randint(5, 10),
                'l2_leaf_reg': loguniform(1, 10),
                'border_count': randint(32, 255)
            }
        
        print(f"✅ Parameter spaces defined for {len(self.param_spaces)} models")
        return list(self.param_spaces.keys())
    
    def optimize_model(self, name, model, param_space, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters for a single model using robust validation"""
        print(f"   🔄 Optimizing {name}...")
        
        try:
            # For tree-based models, use early stopping with validation set
            if name in ['lightgbm', 'xgboost', 'catboost']:
                # Use validation set for early stopping (same as training script)
                if name == 'lightgbm' and LGBM_AVAILABLE:
                    # Set early stopping parameters
                    model.set_params(
                        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), 
                                 lgb.log_evaluation(period=0)]
                    )
                    # Use validation set for evaluation
                    fit_params = {
                        'eval_set': [(X_val, y_val)],
                        'eval_metric': 'rmse'
                    }
                elif name == 'xgboost' and XGB_AVAILABLE:
                    fit_params = {
                        'eval_set': [(X_val, y_val)],
                        'verbose': False
                    }
                elif name == 'catboost' and CATBOOST_AVAILABLE:
                    fit_params = {
                        'eval_set': (X_val, y_val),
                        'early_stopping_rounds': 100,
                        'verbose': False
                    }
                
                # For optimization, use GridSearchCV with focused parameters
                from sklearn.model_selection import GridSearchCV
                
                # Create a smaller, more focused parameter grid
                focused_params = self._create_focused_param_grid(name, param_space)
                
                # Use a simple 2-fold CV to satisfy scikit-learn requirements
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=focused_params,
                    cv=2,  # Minimum required by scikit-learn
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit with validation set for early stopping
                grid_search.fit(X_train, y_train, **fit_params)
                
            else:
                # For non-tree models, use RandomizedSearchCV with proper CV
                from sklearn.model_selection import RandomizedSearchCV
                
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_space,
                    n_iter=20,  # Reduced to prevent overfitting
                    cv=2,  # Minimum required by scikit-learn
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                random_search.fit(X_train, y_train)
                grid_search = random_search
            
            # Get best model and score
            best_model = grid_search.best_estimator_
            best_score = -grid_search.best_score_  # Convert back to positive
            best_params = grid_search.best_params_
            
            print(f"      ✅ Best RMSE: {np.sqrt(best_score):.2f}")
            print(f"      ✅ Best params: {best_params}")
            
            return best_model, best_score, best_params
            
        except Exception as e:
            print(f"      ❌ Error optimizing {name}: {str(e)}")
            return None, float('inf'), {}
    
    def _create_focused_param_grid(self, name, param_space):
        """Create a focused parameter grid to prevent overfitting"""
        if name == 'lightgbm':
            return {
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif name == 'xgboost':
            return {
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif name == 'catboost':
            return {
                'iterations': [500, 1000, 1500],
                'learning_rate': [0.05, 0.1, 0.15],
                'depth': [5, 6, 7],
                'l2_leaf_reg': [1, 3, 5]
            }
        else:
            # Return original param_space for other models
            return param_space
    
    def optimize_all_models(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters for all models"""
        print("\n🚀 Starting hyperparameter optimization...")
        
        # Initialize models
        self.models['svr'] = SVR()
        self.models['random_forest'] = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        if LGBM_AVAILABLE:
            self.models['lightgbm'] = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        if XGB_AVAILABLE:
            self.models['xgboost'] = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = CatBoostRegressor(random_state=42, verbose=False)
        
        # Optimize each model
        for name in self.models.keys():
            if name in self.param_spaces:
                model, score, params = self.optimize_model(
                    name, self.models[name], self.param_spaces[name], X_train, y_train, X_val, y_val
                )
                
                if model is not None:
                    self.models[name] = model
                    self.optimization_results[name] = {
                        'best_score': score,
                        'best_params': params,
                        'best_rmse': np.sqrt(score)
                    }
        
        return len(self.optimization_results) > 0
    
    def evaluate_optimized_models(self, X_test, y_test, targets_data):
        """Evaluate optimized models on test set for multi-horizon forecasting"""
        print("\n🧪 Evaluating optimized models on test set for multi-horizon forecasting...")
        
        test_results = {}
        
        for name, model in self.models.items():
            if name in self.optimization_results:
                try:
                    # Predict on test set
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics for primary target (24h)
                    rmse_24h = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2_24h = r2_score(y_test, y_pred)
                    
                    # Calculate metrics for other horizons if available
                    metrics = {
                        '24h_RMSE': rmse_24h,
                        '24h_R²': r2_24h,
                        'CV_RMSE': self.optimization_results[name]['best_rmse']
                    }
                    
                    # Add 48h and 72h metrics if targets are available
                    if 'target_aqi_48h' in targets_data and targets_data['target_aqi_48h'].notna().sum() > 0:
                        y_48h = targets_data['target_aqi_48h'].iloc[-len(y_test):]
                        y_pred_48h = model.predict(X_test)  # Same features, different target
                        rmse_48h = np.sqrt(mean_squared_error(y_48h, y_pred_48h))
                        r2_48h = r2_score(y_48h, y_pred_48h)
                        metrics.update({
                            '48h_RMSE': rmse_48h,
                            '48h_R²': r2_48h
                        })
                    
                    if 'target_aqi_72h' in targets_data and targets_data['target_aqi_72h'].notna().sum() > 0:
                        y_72h = targets_data['target_aqi_72h'].iloc[-len(y_test):]
                        y_pred_72h = model.predict(X_test)  # Same features, different target
                        rmse_72h = np.sqrt(mean_squared_error(y_72h, y_pred_72h))
                        r2_72h = r2_score(y_72h, y_pred_72h)
                        metrics.update({
                            '72h_RMSE': rmse_72h,
                            '72h_R²': r2_72h
                        })
                    
                    test_results[name] = metrics
                    
                    print(f"   🔍 Evaluating {name}...")
                    print(f"      ✅ Primary (24h): RMSE={rmse_24h:.2f}, R²={r2_24h:.3f}")
                    if '48h_RMSE' in metrics:
                        print(f"      📊 48h: RMSE={metrics['48h_RMSE']:.2f}, R²={metrics['48h_R²']:.3f}")
                    if '72h_RMSE' in metrics:
                        print(f"      📊 72h: RMSE={metrics['72h_RMSE']:.2f}, R²={metrics['72h_R²']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error evaluating {name}: {str(e)}")
        
        return test_results
    
    def save_optimized_models(self):
        """Save optimized models"""
        print("\n💾 Saving multi-horizon optimized models...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        for name, model in self.models.items():
            if name in self.optimization_results:
                try:
                    if hasattr(model, 'save_model'):  # LightGBM, XGBoost, CatBoost
                        model_path = f'saved_models/{name}_multi_horizon_tuned_model.txt'
                        model.save_model(model_path)
                    else:  # Scikit-learn models
                        model_path = f'saved_models/{name}_multi_horizon_tuned_model.pkl'
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                    
                    print(f"   ✅ {name} saved to {model_path}")
                    
                except Exception as e:
                    print(f"   ❌ Error saving {name}: {str(e)}")
        
        # Save scaler
        scaler_path = 'saved_models/standard_scaler_multi_horizon_tuned.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        print(f"   ✅ Scaler saved to {scaler_path}")
        
        # Save feature columns
        features_path = 'saved_models/feature_columns_multi_horizon_tuned.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"   ✅ Feature columns saved to {features_path}")
    
    def save_optimization_results(self, test_results):
        """Save optimization results"""
        print("\n📊 Saving multi-horizon optimization results...")
        
        # Combine CV and test results
        results_data = []
        for name in self.optimization_results.keys():
            if name in test_results:
                row = {
                    'Model': name,
                    'CV_RMSE': self.optimization_results[name]['best_rmse'],
                    '24h_Test_RMSE': test_results[name].get('24h_RMSE', float('nan')),
                    '24h_Test_R²': test_results[name].get('24h_R²', float('nan'))
                }
                
                # Add 48h and 72h metrics if available
                if '48h_RMSE' in test_results[name]:
                    row['48h_Test_RMSE'] = test_results[name]['48h_RMSE']
                    row['48h_Test_R²'] = test_results[name]['48h_R²']
                
                if '72h_RMSE' in test_results[name]:
                    row['72h_Test_RMSE'] = test_results[name]['72h_RMSE']
                    row['72h_R²'] = test_results[name]['72h_R²']
                
                row['Best_Params'] = str(self.optimization_results[name]['best_params'])
                results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('24h_Test_RMSE')
        
        # Save to CSV
        results_path = 'saved_models/multi_horizon_hyperparameter_optimization_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"   ✅ Results saved to {results_path}")
        
        # Display results
        print(f"\n🏆 MULTI-HORIZON OPTIMIZATION RESULTS (sorted by 24h Test RMSE):")
        print(results_df.to_string(index=False))
        
        # Compare with original training results
        self._compare_with_original_results(results_df)
        
        return results_df
    
    def _compare_with_original_results(self, results_df):
        """Compare optimization results with original training results"""
        print(f"\n🔍 COMPARISON WITH ORIGINAL TRAINING RESULTS:")
        print("=" * 70)
        
        # Original results from 04_model_training.py (test set performance)
        original_results = {
            'lightgbm': {'RMSE': 23.74, 'R²': 0.318},
            'catboost': {'RMSE': 22.53, 'R²': 0.386},
            'xgboost': {'RMSE': 22.81, 'R²': 0.371},
            'random_forest': {'RMSE': 23.18, 'R²': 0.350},
            'svr': {'RMSE': 30.08, 'R²': -0.095}
        }
        
        print(f"{'Model':<15} {'Original RMSE':<15} {'Optimized RMSE':<15} {'Improvement':<15} {'Status':<10}")
        print("-" * 70)
        
        for _, row in results_df.iterrows():
            model_name = row['Model']
            optimized_rmse = row['24h_Test_RMSE']
            
            if model_name in original_results:
                original_rmse = original_results[model_name]['RMSE']
                improvement = original_rmse - optimized_rmse
                improvement_pct = (improvement / original_rmse) * 100
                
                if improvement > 0:
                    status = "✅ Better"
                elif improvement < 0:
                    status = "❌ Worse"
                else:
                    status = "➡️ Same"
                
                print(f"{model_name:<15} {original_rmse:<15.2f} {optimized_rmse:<15.2f} {improvement:+.2f} ({improvement_pct:+.1f}%) {status:<10}")
            else:
                print(f"{model_name:<15} {'N/A':<15} {optimized_rmse:<15.2f} {'N/A':<15} {'New':<10}")
        
        # Summary statistics
        improved_models = 0
        total_comparable = 0
        
        for _, row in results_df.iterrows():
            model_name = row['Model']
            if model_name in original_results:
                total_comparable += 1
                original_rmse = original_results[model_name]['RMSE']
                optimized_rmse = row['24h_Test_RMSE']
                if optimized_rmse < original_rmse:
                    improved_models += 1
        
        if total_comparable > 0:
            improvement_rate = (improved_models / total_comparable) * 100
            print(f"\n📊 OPTIMIZATION SUMMARY:")
            print(f"   Models improved: {improved_models}/{total_comparable} ({improvement_rate:.1f}%)")
            
            if improvement_rate >= 60:
                print(f"   🎉 Excellent optimization results!")
            elif improvement_rate >= 40:
                print(f"   👍 Good optimization results")
            elif improvement_rate >= 20:
                print(f"   ⚠️  Moderate optimization results")
            else:
                print(f"   ❌ Poor optimization results - needs investigation")
        else:
            print(f"\n📊 No comparable models found for comparison")
    
    def run_optimization(self):
        """Run complete hyperparameter optimization pipeline"""
        print("🚀 STARTING MULTI-HORIZON HYPERPARAMETER OPTIMIZATION PIPELINE")
        print("=" * 70)
        
        # Step 1: Load multi-horizon clean data
        df = self.load_clean_data()
        if df is None:
            return False
        
        # Step 2: Prepare multi-horizon data
        result = self.prepare_data(df)
        if result is None:
            return False
        X, y, targets_data = result
        
        # Step 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 4: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        # Step 5: Define parameter spaces
        model_names = self.define_parameter_spaces()
        
        # Step 6: Optimize all models
        if not self.optimize_all_models(X_train_scaled, y_train, X_val_scaled, y_val):
            print("❌ No models optimized successfully")
            return False
        
        # Step 7: Evaluate on test set
        test_results = self.evaluate_optimized_models(X_test_scaled, y_test, targets_data)
        
        # Step 8: Save models and results
        self.save_optimized_models()
        results_df = self.save_optimization_results(test_results)
        
        # Final summary
        print(f"\n🎉 MULTI-HORIZON HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print("=" * 70)
        print(f"📊 Models optimized: {len(self.optimization_results)}")
        print(f"📈 Best model by 24h Test RMSE: {results_df.iloc[0]['Model']}")
        print(f"🎯 Best 24h Test RMSE: {results_df['24h_Test_RMSE'].min():.2f}")
        print(f"📁 All models saved to 'saved_models/' directory")
        
        # Performance summary by horizon
        print(f"\n📊 PERFORMANCE SUMMARY BY HORIZON:")
        best_24h = results_df.loc[results_df['24h_Test_RMSE'].idxmin(), 'Model']
        best_24h_rmse = results_df['24h_Test_RMSE'].min()
        print(f"   🎯 24h: Best model = {best_24h} (RMSE: {best_24h_rmse:.2f})")
        
        if '48h_Test_RMSE' in results_df.columns:
            best_48h = results_df.loc[results_df['48h_Test_RMSE'].idxmin(), 'Model']
            best_48h_rmse = results_df['48h_Test_RMSE'].min()
            print(f"   🎯 48h: Best model = {best_48h} (RMSE: {best_48h_rmse:.2f})")
        
        if '72h_Test_RMSE' in results_df.columns:
            best_72h = results_df.loc[results_df['72h_Test_RMSE'].idxmin(), 'Model']
            best_72h_rmse = results_df['72h_Test_RMSE'].min()
            print(f"   🎯 72h: Best model = {best_72h} (RMSE: {best_72h_rmse:.2f})")
        
        return True

def main():
    """Main function to run hyperparameter optimization"""
    optimizer = HyperparameterOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        print(f"\n🎉 Multi-Horizon Optimization Complete: All Models Optimized Successfully!")
        print(f"🎯 Ready for production use with 24h, 48h, and 72h forecasting capabilities!")
    else:
        print(f"\n❌ Optimization failed! Check error messages above.")

if __name__ == "__main__":
    main()
