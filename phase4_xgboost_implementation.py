"""
Phase 4: XGBoost Implementation
==============================

Implement XGBoost with hyperparameter tuning to push beyond the 77.4% R² baseline.
Target: Optimize for even better performance and robust forecasting.

Current status: 77.4% R² baseline (already above 75% target!)
Goal: Push towards 80%+ R² with XGBoost optimization
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
from typing import Dict, List, Tuple

# XGBoost and optimization
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load the Phase 4 foundation
from phase4_model_development import Phase4ModelDevelopment

warnings.filterwarnings('ignore')

class XGBoostOptimizer:
    """
    XGBoost Implementation and Optimization for AQI Prediction
    Goal: Push beyond 77.4% R² baseline towards 80%+
    """
    
    def __init__(self, phase4_base: Phase4ModelDevelopment):
        """Initialize with Phase 4 foundation"""
        self.base = phase4_base
        self.models = {}
        self.results = {}
        
        print("🚀 XGBOOST OPTIMIZATION")
        print("=" * 30)
        print(f"📊 Current baseline: {phase4_base.results['random_forest']['test_r2']:.3f} R²")
        print("🎯 Goal: Push towards 80%+ R²")

    def step4_basic_xgboost(self):
        """Step 4: Basic XGBoost implementation"""
        print("\n🚀 STEP 4: Basic XGBoost Implementation")
        print("-" * 38)
        
        try:
            print("⚙️  Training basic XGBoost model...")
            
            # Basic XGBoost configuration
            xgb_basic = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            # Train model
            xgb_basic.fit(
                self.base.X_train, 
                self.base.y_train,
                eval_set=[(self.base.X_train, self.base.y_train), (self.base.X_test, self.base.y_test)],
                eval_metric='rmse',
                verbose=False
            )
            
            # Predictions
            y_pred_train = xgb_basic.predict(self.base.X_train)
            y_pred_test = xgb_basic.predict(self.base.X_test)
            
            # Metrics
            train_r2 = r2_score(self.base.y_train, y_pred_train)
            test_r2 = r2_score(self.base.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.base.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.base.y_test, y_pred_test))
            
            print(f"📊 Basic XGBoost Results:")
            print(f"   Training R²: {train_r2:.3f}")
            print(f"   Test R²: {test_r2:.3f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test RMSE: {test_rmse:.2f}")
            
            # Compare with baseline
            rf_r2 = self.base.results['random_forest']['test_r2']
            improvement = test_r2 - rf_r2
            print(f"💡 Improvement over RF: {improvement:+.3f} R² ({improvement/rf_r2*100:+.1f}%)")
            
            # Store results
            self.models['xgboost_basic'] = xgb_basic
            self.results['xgboost_basic'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'model_type': 'XGBoost Basic',
                'improvement_vs_rf': improvement
            }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.base.feature_names,
                'importance': xgb_basic.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"🔝 Top 5 XGBoost features:")
            for i, row in feature_importance.head().iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in basic XGBoost: {str(e)}")
            return False

    def step5_hyperparameter_tuning(self):
        """Step 5: XGBoost hyperparameter tuning"""
        print("\n🔧 STEP 5: XGBoost Hyperparameter Tuning")
        print("-" * 41)
        
        try:
            print("🔍 Starting hyperparameter optimization...")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5, test_size=200)
            
            # Hyperparameter search space
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
            
            # Randomized search for efficiency
            print("⚡ Using RandomizedSearchCV for efficiency...")
            
            xgb_regressor = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            random_search = RandomizedSearchCV(
                estimator=xgb_regressor,
                param_distributions=param_grid,
                n_iter=50,  # Reasonable number of iterations
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            # Fit the search
            print("🔄 Training 50 XGBoost configurations...")
            random_search.fit(self.base.X_train, self.base.y_train)
            
            # Best model
            best_xgb = random_search.best_estimator_
            
            print(f"✅ Hyperparameter tuning completed!")
            print(f"🏆 Best CV score: {random_search.best_score_:.3f}")
            print(f"⚙️  Best parameters:")
            for param, value in random_search.best_params_.items():
                print(f"   {param}: {value}")
            
            # Evaluate on test set
            y_pred_train = best_xgb.predict(self.base.X_train)
            y_pred_test = best_xgb.predict(self.base.X_test)
            
            train_r2 = r2_score(self.base.y_train, y_pred_train)
            test_r2 = r2_score(self.base.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.base.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.base.y_test, y_pred_test))
            
            print(f"\n📊 Optimized XGBoost Results:")
            print(f"   Training R²: {train_r2:.3f}")
            print(f"   Test R²: {test_r2:.3f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test RMSE: {test_rmse:.2f}")
            
            # Improvements
            basic_r2 = self.results['xgboost_basic']['test_r2']
            rf_r2 = self.base.results['random_forest']['test_r2']
            
            improvement_vs_basic = test_r2 - basic_r2
            improvement_vs_rf = test_r2 - rf_r2
            
            print(f"💡 Improvement over basic XGBoost: {improvement_vs_basic:+.3f} R²")
            print(f"💡 Improvement over Random Forest: {improvement_vs_rf:+.3f} R²")
            
            # Store results
            self.models['xgboost_optimized'] = best_xgb
            self.results['xgboost_optimized'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'model_type': 'XGBoost Optimized',
                'best_params': random_search.best_params_,
                'cv_score': random_search.best_score_,
                'improvement_vs_basic': improvement_vs_basic,
                'improvement_vs_rf': improvement_vs_rf
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error in hyperparameter tuning: {str(e)}")
            return False

    def step6_feature_importance_analysis(self):
        """Step 6: Detailed feature importance analysis"""
        print("\n📊 STEP 6: Feature Importance Analysis")
        print("-" * 38)
        
        try:
            best_model = self.models['xgboost_optimized']
            
            # Get feature importance
            importance_gain = best_model.feature_importances_
            
            # Create comprehensive importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.base.feature_names,
                'importance_gain': importance_gain,
                'importance_gain_normalized': importance_gain / importance_gain.sum()
            }).sort_values('importance_gain', ascending=False)
            
            # Save detailed importance
            importance_file = "data_repositories/features/phase4_xgboost_feature_importance.csv"
            feature_importance_df.to_csv(importance_file, index=False)
            
            print(f"📁 Feature importance saved: {importance_file}")
            print(f"\n🔝 Top 10 Most Important Features:")
            
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"   {i+1:2d}. {row['feature']:<35} {row['importance_gain']:.4f} ({row['importance_gain_normalized']*100:.1f}%)")
            
            # Feature categories analysis
            print(f"\n📋 Feature Importance by Category:")
            
            categories = {
                'pollution': ['pm2_5', 'pm10', 'no2', 'o3', 'aqi'],
                'weather': ['temperature', 'humidity', 'wind', 'pressure'],
                'temporal': ['hour', 'day', 'month', 'season', 'weekend'],
                'lag': ['lag'],
                'rolling': ['rolling']
            }
            
            for category, keywords in categories.items():
                category_features = feature_importance_df[
                    feature_importance_df['feature'].str.contains('|'.join(keywords), case=False)
                ]
                total_importance = category_features['importance_gain_normalized'].sum()
                print(f"   {category:.<15} {total_importance*100:>5.1f}% ({len(category_features)} features)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in feature importance analysis: {str(e)}")
            return False

    def run_xgboost_optimization(self):
        """Run complete XGBoost optimization pipeline"""
        print("\n🚀 EXECUTING XGBOOST OPTIMIZATION")
        print("=" * 40)
        
        steps = [
            ("Basic XGBoost", self.step4_basic_xgboost),
            ("Hyperparameter Tuning", self.step5_hyperparameter_tuning),
            ("Feature Importance", self.step6_feature_importance_analysis)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ STEP FAILED: {step_name}")
                return False
            print(f"✅ STEP COMPLETED: {step_name}")
        
        # Final summary
        print(f"\n🎉 XGBOOST OPTIMIZATION COMPLETED!")
        print("=" * 35)
        
        best_r2 = self.results['xgboost_optimized']['test_r2']
        target_r2 = 0.75
        
        print(f"🏆 Best XGBoost R²: {best_r2:.3f}")
        print(f"🎯 Target R²: {target_r2:.3f}")
        print(f"📈 Exceeded target by: {best_r2 - target_r2:+.3f} R²")
        
        # Save final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 4 - XGBoost Optimization',
            'target_achieved': best_r2 >= target_r2,
            'best_model': 'xgboost_optimized',
            'performance': self.results,
            'summary': {
                'best_r2': best_r2,
                'target_r2': target_r2,
                'target_exceeded': best_r2 - target_r2,
                'models_tested': list(self.results.keys())
            }
        }
        
        results_file = "data_repositories/features/phase4_xgboost_final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4, default=str)
        
        print(f"💾 Final results saved: {results_file}")
        
        return True

def main():
    """Run XGBoost optimization"""
    print("🔄 Loading Phase 4 foundation...")
    
    # Initialize Phase 4 base
    phase4_base = Phase4ModelDevelopment()
    if not phase4_base.run_phase4_step1_to_3():
        print("❌ Phase 4 foundation failed")
        return
    
    # Run XGBoost optimization
    xgb_optimizer = XGBoostOptimizer(phase4_base)
    success = xgb_optimizer.run_xgboost_optimization()
    
    if success:
        print("\n🏆 XGBOOST OPTIMIZATION SUCCESS!")
        print("Ready for LightGBM and Neural Networks")
    else:
        print("\n⚠️  XGBOOST OPTIMIZATION INCOMPLETE")

if __name__ == "__main__":
    main()
