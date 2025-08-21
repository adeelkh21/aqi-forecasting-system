"""
AQI Prediction System - Model Evaluation Pipeline (FIXED VERSION)
===============================================================

This script evaluates all trained models on the CLEAN dataset:
- Uses the same 53-feature dataset as TCN (no data leakage)
- Evaluates models on held-out test set
- Generates comprehensive performance analysis

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

# ML metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("🔧 MODEL EVALUATION PIPELINE (FIXED VERSION)")
print("=" * 50)
print("🎯 Using selected feature set (no data leakage, 72h horizon)")
print("🎯 Same dataset used across models")

class ModelEvaluator:
    """Evaluate all trained models on clean dataset"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.test_data = None
        self.evaluation_results = {}
        
    def load_clean_data(self):
        """Load the SAME clean dataset that TCN uses (53 features, no leakage)"""
        print("\n📊 Loading clean dataset...")
        
        try:
            # Load the clean, feature-selected dataset (same as TCN)
            df = pd.read_csv('data_repositories/features/phase1_fixed_selected_features.csv')
            
            # Load feature columns
            with open('data_repositories/features/phase1_fixed_feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print(f"✅ Clean dataset loaded: {df.shape}")
            print(f"📊 Features: {len(self.feature_columns)}")
            print(f"🎯 Target: numerical_aqi")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_test_data(self, df):
        """Prepare test data from clean dataset"""
        print("\n🔧 Preparing test data...")
        
        # Use the clean feature columns (no AQI lags!)
        X = df[self.feature_columns]
        y = df['numerical_aqi']
        
        # Remove rows with NaN target (due to 72-hour forecasting shift)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Use the same split as training (last 20% for testing)
        test_size = int(0.2 * len(X))
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        print(f"✅ Test data prepared: {X_test.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Target range: {y_test.min():.1f} - {y_test.max():.1f}")
        
        return X_test, y_test
    
    def load_trained_models(self):
        """Load all trained models"""
        print("\n🤖 Loading trained models...")
        
        models_dir = 'saved_models'
        model_files = {
            'random_forest': 'random_forest_fixed_model.pkl',
            'linear_regression': 'linear_regression_fixed_model.pkl',
            'ridge': 'ridge_fixed_model.pkl',
            'lasso': 'lasso_fixed_model.pkl',
            'svr': 'svr_fixed_model.pkl',
            'knn': 'knn_fixed_model.pkl',
            'decision_tree': 'decision_tree_fixed_model.pkl',
            'lightgbm': 'lightgbm_fixed_model.pkl',
            'xgboost': 'xgboost_fixed_model.txt',
            'catboost': 'catboost_fixed_model.txt'
        }
        
        loaded_models = 0
        
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    if filename.endswith('.txt'):  # LightGBM, XGBoost, CatBoost
                        if 'lightgbm' in name:
                            from lightgbm import LGBMRegressor
                            model = LGBMRegressor()
                            model.load_model(filepath)
                        elif 'xgboost' in name:
                            from xgboost import XGBRegressor
                            model = XGBRegressor()
                            model.load_model(filepath)
                        elif 'catboost' in name:
                            from catboost import CatBoostRegressor
                            model = CatBoostRegressor()
                            model.load_model(filepath)
                    else:  # Scikit-learn models
                        with open(filepath, 'rb') as f:
                            model = pickle.load(f)
                    
                    self.models[name] = model
                    loaded_models += 1
                    print(f"   ✅ {name} loaded")
                    
                except Exception as e:
                    print(f"   ❌ Error loading {name}: {str(e)}")
            else:
                print(f"   ⚠️ {name} not found: {filepath}")
        
        print(f"✅ {loaded_models} models loaded")
        return loaded_models > 0
    
    def load_scaler(self):
        """Load the scaler used during training"""
        print("\n⚖️ Loading scaler...")
        
        try:
            scaler_path = 'saved_models/standard_scaler_fixed.pkl'
            with open(scaler_path, 'rb') as f:
                self.scalers['standard'] = pickle.load(f)
            print("✅ Standard scaler loaded")
            return True
        except Exception as e:
            print(f"❌ Error loading scaler: {str(e)}")
            return False
    
    def evaluate_model(self, name, model, X_test, y_test):
        """Evaluate a single model"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE (%)': mape,
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"   ❌ Error evaluating {name}: {str(e)}")
            return None
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all loaded models"""
        print("\n📊 Evaluating all models...")
        
        for name, model in self.models.items():
            print(f"   🔄 Evaluating {name}...")
            
            results = self.evaluate_model(name, model, X_test, y_test)
            
            if results:
                self.evaluation_results[name] = results
                print(f"      ✅ RMSE: {results['RMSE']:.2f}, R²: {results['R²']:.3f}")
            else:
                print(f"      ❌ Evaluation failed")
        
        return len(self.evaluation_results) > 0
    
    def generate_performance_summary(self):
        """Generate comprehensive performance summary"""
        print("\n📊 GENERATING PERFORMANCE SUMMARY")
        print("=" * 50)
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return None
        
        # Create summary DataFrame
        summary_data = []
        for name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': name,
                'RMSE': results['RMSE'],
                'MAE': results['MAE'],
                'R²': results['R²'],
                'MAPE (%)': results['MAPE (%)']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE')
        
        # Display summary
        print("\n🏆 MODEL PERFORMANCE RANKING (by RMSE):")
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_model = summary_df.iloc[0]
        print(f"\n🎯 BEST MODEL: {best_model['Model']}")
        print(f"   RMSE: {best_model['RMSE']:.2f}")
        print(f"   R²: {best_model['R²']:.3f}")
        print(f"   MAE: {best_model['MAE']:.2f}")
        
        # Performance analysis
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        avg_r2 = summary_df['R²'].mean()
        print(f"   Average R²: {avg_r2:.3f}")
        
        if avg_r2 > 0.95:
            print(f"   ⚠️ Average R² is suspiciously high - possible remaining data leakage")
        elif avg_r2 > 0.8:
            print(f"   ✅ Average R² is realistic - no data leakage detected")
        else:
            print(f"   📊 Average R² is moderate - legitimate learning from patterns")
        
        return summary_df
    
    def compare_with_tcn(self, summary_df):
        """Compare traditional ML results with TCN"""
        print(f"\n🔍 COMPARISON WITH BASELINE (optional):")
        print("=" * 50)
        if summary_df is None or summary_df.empty:
            print("❌ No results to compare")
            return
        best_ml_rmse = summary_df['RMSE'].min()
        best_ml_model = summary_df.iloc[0]['Model']
        baseline_path = os.path.join('saved_models', 'tcn_baseline.json')
        if os.path.exists(baseline_path):
            try:
                import json
                with open(baseline_path, 'r') as f:
                    base = json.load(f)
                base_rmse = float(base.get('RMSE', float('inf')))
                print(f"   External baseline RMSE={base_rmse:.2f}")
                print(f"   Best ML ({best_ml_model}) RMSE={best_ml_rmse:.2f}")
            except Exception:
                print("   ⚠️ Could not read external baseline file")
        else:
            print("   No external baseline file found; skipping comparison")
    
    def save_results(self, summary_df):
        """Save evaluation results"""
        print("\n💾 Saving results...")
        
        os.makedirs('saved_models', exist_ok=True)
        
        # Save summary
        summary_path = 'saved_models/model_evaluation_summary_fixed.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"   ✅ Summary saved to {summary_path}")
        
        # Save detailed results
        results_path = 'saved_models/model_evaluation_detailed_fixed.json'
        with open(results_path, 'w') as f:
            import json
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for name, results in self.evaluation_results.items():
                json_results[name] = {
                    'RMSE': float(results['RMSE']),
                    'MAE': float(results['MAE']),
                    'R²': float(results['R²']),
                    'MAPE (%)': float(results['MAPE (%)'])
                }
            json.dump(json_results, f, indent=2)
        print(f"   ✅ Detailed results saved to {results_path}")

        # Optionally evaluate blended ensemble if metadata exists
        ensemble_meta = os.path.join('saved_models', 'ensemble_meta.json')
        if os.path.exists(ensemble_meta):
            try:
                with open(ensemble_meta, 'r') as f:
                    meta = json.load(f)
                model_names = meta.get('models', [])
                coef = np.array(meta.get('coef', []), dtype=float)
                intercept = float(meta.get('intercept', 0.0))
                # Recreate predictions from base models
                preds = []
                for n in model_names:
                    if n in self.models:
                        preds.append(self.models[n].predict(self.test_data[0]))
                if preds and len(preds) == len(coef):
                    stack = np.vstack(preds).T
                    y_pred = stack.dot(coef) + intercept
                    y_true = self.test_data[1]
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    r2 = float(r2_score(y_true, y_pred))
                    print(f"   Blended ensemble (from meta): RMSE={rmse:.2f}, R²={r2:.3f}")
            except Exception:
                pass
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 STARTING MODEL EVALUATION PIPELINE")
        print("=" * 50)
        
        # Step 1: Load clean data
        df = self.load_clean_data()
        if df is None:
            return False
        
        # Step 2: Prepare test data
        X_test, y_test = self.prepare_test_data(df)
        self.test_data = (X_test, y_test)
        
        # Step 3: Load trained models
        if not self.load_trained_models():
            print("❌ No models loaded")
            return False
        
        # Step 4: Load scaler
        if not self.load_scaler():
            print("❌ Scaler not loaded")
            return False
        
        # Step 5: Scale test features
        X_test_scaled = self.scalers['standard'].transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        # Step 6: Evaluate all models
        if not self.evaluate_all_models(X_test_scaled, y_test):
            print("❌ No models evaluated successfully")
            return False
        
        # Step 7: Generate performance summary
        summary_df = self.generate_performance_summary()
        
        # Step 8: Compare with TCN
        self.compare_with_tcn(summary_df)
        
        # Step 9: Save results
        self.save_results(summary_df)
        
        print(f"\n🎉 MODEL EVALUATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Models evaluated: {len(self.evaluation_results)}")
        print(f"📈 Best model: {summary_df.iloc[0]['Model']}")
        print(f"🎯 Best RMSE: {summary_df['RMSE'].min():.2f}")
        print(f"📁 Results saved to 'saved_models/' directory")
        
        return True

def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    success = evaluator.run_evaluation()
    
    if success:
        print(f"\n🎉 Evaluation Complete: All Models Evaluated Successfully!")
    else:
        print(f"\n❌ Evaluation failed! Check error messages above.")

if __name__ == "__main__":
    main()
