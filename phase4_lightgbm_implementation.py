"""
Phase 4: LightGBM Implementation
===============================

Implement LightGBM to compete with XGBoost's 89.3% R² performance.
Goal: See if LightGBM can match or exceed XGBoost performance.

Current Status:
- Random Forest: 77.4% R²
- XGBoost: 89.3% R² (Target exceeded by +14.3%)
- Target: 75% R² ✅ ACHIEVED!

Next Goal: LightGBM optimization for potential 90%+ R²
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
from typing import Dict, List, Tuple

# LightGBM and optimization
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load the Phase 4 foundation
from phase4_model_development import Phase4ModelDevelopment

warnings.filterwarnings('ignore')

class LightGBMOptimizer:
    """
    LightGBM Implementation and Optimization for AQI Prediction
    Goal: Compete with XGBoost's 89.3% R² performance
    """
    
    def __init__(self, phase4_base: Phase4ModelDevelopment):
        """Initialize with Phase 4 foundation"""
        self.base = phase4_base
        self.models = {}
        self.results = {}
        
        print("⚡ LIGHTGBM OPTIMIZATION")
        print("=" * 30)
        print(f"🏆 XGBoost benchmark: 89.3% R²")
        print("🎯 Goal: Match or exceed XGBoost performance")

    def step7_basic_lightgbm(self):
        """Step 7: Basic LightGBM implementation"""
        print("\n⚡ STEP 7: Basic LightGBM Implementation")
        print("-" * 39)
        
        try:
            print("⚙️  Training basic LightGBM model...")
            
            # Basic LightGBM configuration
            lgb_basic = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            # Train model
            lgb_basic.fit(
                self.base.X_train, 
                self.base.y_train,
                eval_set=[(self.base.X_train, self.base.y_train), (self.base.X_test, self.base.y_test)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predictions
            y_pred_train = lgb_basic.predict(self.base.X_train)
            y_pred_test = lgb_basic.predict(self.base.X_test)
            
            # Metrics
            train_r2 = r2_score(self.base.y_train, y_pred_train)
            test_r2 = r2_score(self.base.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.base.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.base.y_test, y_pred_test))
            
            print(f"📊 Basic LightGBM Results:")
            print(f"   Training R²: {train_r2:.3f}")
            print(f"   Test R²: {test_r2:.3f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test RMSE: {test_rmse:.2f}")
            
            # Compare with XGBoost
            xgb_r2 = 0.893  # From previous results
            improvement = test_r2 - xgb_r2
            print(f"💡 vs XGBoost: {improvement:+.3f} R² ({improvement/xgb_r2*100:+.1f}%)")
            
            # Store results
            self.models['lightgbm_basic'] = lgb_basic
            self.results['lightgbm_basic'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'model_type': 'LightGBM Basic',
                'vs_xgboost': improvement
            }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.base.feature_names,
                'importance': lgb_basic.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"🔝 Top 5 LightGBM features:")
            for i, row in feature_importance.head().iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.0f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in basic LightGBM: {str(e)}")
            return False

    def step8_lightgbm_hyperparameter_tuning(self):
        """Step 8: LightGBM hyperparameter tuning"""
        print("\n🔧 STEP 8: LightGBM Hyperparameter Tuning")
        print("-" * 42)
        
        try:
            print("🔍 Starting LightGBM hyperparameter optimization...")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5, test_size=200)
            
            # LightGBM-specific hyperparameter space
            param_grid = {
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'feature_fraction': [0.6, 0.7, 0.8, 0.9],
                'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
                'bagging_freq': [1, 3, 5, 7],
                'min_child_samples': [10, 20, 30, 50],
                'n_estimators': [200, 300, 500],
                'reg_alpha': [0, 0.1, 0.3, 0.5],
                'reg_lambda': [0, 0.1, 0.3, 0.5]
            }
            
            # Randomized search for efficiency
            print("⚡ Using RandomizedSearchCV for efficiency...")
            
            lgb_regressor = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            random_search = RandomizedSearchCV(
                estimator=lgb_regressor,
                param_distributions=param_grid,
                n_iter=50,  # 50 iterations like XGBoost
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            # Fit the search
            print("🔄 Training 50 LightGBM configurations...")
            random_search.fit(self.base.X_train, self.base.y_train)
            
            # Best model
            best_lgb = random_search.best_estimator_
            
            print(f"✅ Hyperparameter tuning completed!")
            print(f"🏆 Best CV score: {random_search.best_score_:.3f}")
            print(f"⚙️  Best parameters:")
            for param, value in random_search.best_params_.items():
                print(f"   {param}: {value}")
            
            # Evaluate on test set
            y_pred_train = best_lgb.predict(self.base.X_train)
            y_pred_test = best_lgb.predict(self.base.X_test)
            
            train_r2 = r2_score(self.base.y_train, y_pred_train)
            test_r2 = r2_score(self.base.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.base.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.base.y_test, y_pred_test))
            
            print(f"\n📊 Optimized LightGBM Results:")
            print(f"   Training R²: {train_r2:.3f}")
            print(f"   Test R²: {test_r2:.3f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test RMSE: {test_rmse:.2f}")
            
            # Improvements
            basic_r2 = self.results['lightgbm_basic']['test_r2']
            xgb_r2 = 0.893  # XGBoost benchmark
            
            improvement_vs_basic = test_r2 - basic_r2
            improvement_vs_xgb = test_r2 - xgb_r2
            
            print(f"💡 Improvement over basic LightGBM: {improvement_vs_basic:+.3f} R²")
            print(f"💡 vs XGBoost benchmark: {improvement_vs_xgb:+.3f} R²")
            
            # Store results
            self.models['lightgbm_optimized'] = best_lgb
            self.results['lightgbm_optimized'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'model_type': 'LightGBM Optimized',
                'best_params': random_search.best_params_,
                'cv_score': random_search.best_score_,
                'improvement_vs_basic': improvement_vs_basic,
                'vs_xgboost': improvement_vs_xgb
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error in LightGBM hyperparameter tuning: {str(e)}")
            return False

    def step9_model_comparison(self):
        """Step 9: Detailed model comparison"""
        print("\n🏆 STEP 9: Model Performance Comparison")
        print("-" * 40)
        
        try:
            # Performance summary
            models_performance = {
                'Random Forest': 0.774,
                'XGBoost': 0.893,
                'LightGBM Basic': self.results['lightgbm_basic']['test_r2'],
                'LightGBM Optimized': self.results['lightgbm_optimized']['test_r2']
            }
            
            print("📊 Performance Leaderboard:")
            print("-" * 35)
            sorted_models = sorted(models_performance.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (model, r2) in enumerate(sorted_models, 1):
                target_gap = r2 - 0.75
                print(f"{rank}. {model:<20} {r2:.3f} R² (Target: {target_gap:+.3f})")
            
            # Best model identification
            best_model_name = sorted_models[0][0]
            best_r2 = sorted_models[0][1]
            
            print(f"\n🏆 CHAMPION MODEL: {best_model_name}")
            print(f"🎯 Performance: {best_r2:.3f} R²")
            print(f"📈 Target exceeded by: {best_r2 - 0.75:+.3f} R²")
            
            # Feature importance comparison
            if 'lightgbm_optimized' in self.models:
                lgb_model = self.models['lightgbm_optimized']
                feature_importance = pd.DataFrame({
                    'feature': self.base.feature_names,
                    'lgb_importance': lgb_model.feature_importances_
                }).sort_values('lgb_importance', ascending=False)
                
                print(f"\n🔝 Top 5 LightGBM features:")
                for i, row in feature_importance.head().iterrows():
                    print(f"   {i+1}. {row['feature']}: {row['lgb_importance']:.0f}")
            
            # Save comparison results
            comparison_results = {
                'timestamp': datetime.now().isoformat(),
                'leaderboard': dict(sorted_models),
                'champion': {
                    'model': best_model_name,
                    'r2_score': best_r2,
                    'target_exceeded': best_r2 - 0.75
                },
                'target_achievement': {
                    'target_r2': 0.75,
                    'achieved': best_r2 >= 0.75,
                    'all_models_above_target': all(r2 >= 0.75 for r2 in models_performance.values())
                }
            }
            
            comparison_file = "data_repositories/features/phase4_model_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=4, default=str)
            
            print(f"💾 Comparison saved: {comparison_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in model comparison: {str(e)}")
            return False

    def run_lightgbm_optimization(self):
        """Run complete LightGBM optimization pipeline"""
        print("\n⚡ EXECUTING LIGHTGBM OPTIMIZATION")
        print("=" * 40)
        
        steps = [
            ("Basic LightGBM", self.step7_basic_lightgbm),
            ("Hyperparameter Tuning", self.step8_lightgbm_hyperparameter_tuning),
            ("Model Comparison", self.step9_model_comparison)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ STEP FAILED: {step_name}")
                return False
            print(f"✅ STEP COMPLETED: {step_name}")
        
        # Final summary
        print(f"\n🎉 LIGHTGBM OPTIMIZATION COMPLETED!")
        print("=" * 36)
        
        best_r2 = self.results['lightgbm_optimized']['test_r2']
        target_r2 = 0.75
        xgb_r2 = 0.893
        
        print(f"⚡ Best LightGBM R²: {best_r2:.3f}")
        print(f"🏆 XGBoost R²: {xgb_r2:.3f}")
        print(f"🎯 Target R²: {target_r2:.3f}")
        
        # Winner determination
        if best_r2 > xgb_r2:
            print(f"🏆 LIGHTGBM WINS! (+{best_r2 - xgb_r2:.3f} R²)")
        elif best_r2 < xgb_r2:
            print(f"🏆 XGBOOST WINS! (+{xgb_r2 - best_r2:.3f} R²)")
        else:
            print(f"🤝 TIE! Both models achieved {best_r2:.3f} R²")
        
        # Save final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 4 - LightGBM Optimization',
            'target_achieved': best_r2 >= target_r2,
            'champion_model': 'LightGBM' if best_r2 >= xgb_r2 else 'XGBoost',
            'performance': self.results,
            'summary': {
                'lightgbm_r2': best_r2,
                'xgboost_r2': xgb_r2,
                'target_r2': target_r2,
                'target_exceeded': max(best_r2, xgb_r2) - target_r2,
                'models_tested': list(self.results.keys())
            }
        }
        
        results_file = "data_repositories/features/phase4_lightgbm_final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4, default=str)
        
        print(f"💾 Final results saved: {results_file}")
        
        return True

def main():
    """Run LightGBM optimization"""
    print("🔄 Loading Phase 4 foundation...")
    
    # Initialize Phase 4 base
    phase4_base = Phase4ModelDevelopment()
    if not phase4_base.run_phase4_step1_to_3():
        print("❌ Phase 4 foundation failed")
        return
    
    # Run LightGBM optimization
    lgb_optimizer = LightGBMOptimizer(phase4_base)
    success = lgb_optimizer.run_lightgbm_optimization()
    
    if success:
        print("\n🏆 LIGHTGBM OPTIMIZATION SUCCESS!")
        print("Ready for ensemble methods and final evaluation")
    else:
        print("\n⚠️  LIGHTGBM OPTIMIZATION INCOMPLETE")

if __name__ == "__main__":
    main()
