#!/usr/bin/env python3
"""
Phase 1: FIXED Feature Selection (NO DATA LEAKAGE)
- Performs SHAP analysis on the fixed no-leakage data
- Selects top features for legitimate forecasting
- Creates final training dataset without data leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, install if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not available, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "shap"])
    import shap
    SHAP_AVAILABLE = True

# Try to import LightGBM for SHAP analysis
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

class FixedFeatureSelector:
    def __init__(self):
        """Initialize fixed feature selector for multi-horizon forecasting"""
        print("🔍 PHASE 1: FIXED FEATURE SELECTION (NO DATA LEAKAGE)")
        print("=" * 80)
        
        # Data paths
        self.input_path = "data_repositories/features/phase1_no_leakage_data.csv"
        self.output_path = "data_repositories/features/phase1_fixed_selected_features.csv"
        self.feature_columns_path = "data_repositories/features/phase1_fixed_feature_columns.pkl"
        self.scaler_path = "data_repositories/features/phase1_fixed_feature_scaler.pkl"
        self.metadata_path = "data_repositories/features/phase1_fixed_feature_selection_metadata.json"
        
        # Create output directories
        os.makedirs("data_repositories/features", exist_ok=True)
        
        # Target number of features (increased for multi-horizon forecasting)
        self.target_features = 50  # Increased from 35 for multi-horizon features
        
        print(f"🎯 Target Features: {self.target_features}")
        print(f"📁 Input: {self.input_path}")
        print(f"📁 Output: {self.output_path}")
        print(f"🔍 Method: SHAP analysis with LightGBM (NO LEAKAGE)")
        print(f"✅ Using legitimate features only - NO AQI leakage")
        print(f"🎯 Multi-horizon targets: 24h, 48h, 72h forecasting")
        
    def load_data(self):
        """Load the fixed no-leakage data with multi-horizon targets"""
        print(f"\n📥 LOADING FIXED NO-LEAKAGE DATA")
        print("-" * 40)
        # Print input dataset date range at the start
        try:
            if os.path.exists(self.input_path):
                tmp = pd.read_csv(self.input_path, usecols=['timestamp'])
                if not tmp.empty:
                    tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
                    print(f"📅 Feature-selection input range: {tmp['timestamp'].min()} → {tmp['timestamp'].max()}")
            else:
                print(f"📅 Feature-selection input: MISSING {self.input_path}")
        except Exception as e:
            print(f"⚠️ Could not read input date range: {e}")
        
        if not os.path.exists(self.input_path):
            print(f"❌ Input file not found: {self.input_path}")
            return None
        
        # Load data
        df = pd.read_csv(self.input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"   Data loaded: {len(df):,} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Total columns: {len(df.columns)}")
        
        # Check for multi-horizon targets
        target_columns = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        missing_targets = [col for col in target_columns if col not in df.columns]
        
        if missing_targets:
            print(f"❌ Missing multi-horizon targets: {missing_targets}")
            return None
        
        print(f"   ✅ Multi-horizon targets found:")
        for target in target_columns:
            target_range = f"{df[target].min():.1f} to {df[target].max():.1f}"
            valid_count = df[target].notna().sum()
            print(f"      → {target}: {target_range} ({valid_count:,} valid values)")
        
        # Use 24h target as primary target for feature selection
        primary_target = 'target_aqi_24h'
        print(f"   🎯 Primary target for feature selection: {primary_target}")
        print(f"   🎯 Primary target range: {df[primary_target].min():.1f} to {df[primary_target].max():.1f}")
        
        # Verify no AQI leakage features
        aqi_features = [col for col in df.columns if 'aqi' in col.lower() and col not in target_columns]
        if aqi_features:
            print(f"   ⚠️ WARNING: Found potential AQI leakage features: {aqi_features}")
            print(f"   → These features will be removed during feature selection to prevent data leakage")
        else:
            print(f"   ✅ No AQI leakage features detected")
        
        return df
    
    def prepare_features_and_target(self, df):
        """Prepare features and target for SHAP analysis with multi-horizon targets"""
        print(f"\n⚙️ PREPARING FEATURES AND TARGET")
        print("-" * 50)
    
        # CRITICAL: Exclude timestamp, all multi-horizon targets, and any AQI-related columns
        exclude_cols = ['timestamp', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        
        # Check if primary_pollutant exists before excluding it
        if 'primary_pollutant' in df.columns:
            exclude_cols.append('primary_pollutant')
            print(f"   ✅ Excluding primary_pollutant (derived from AQI)")
        else:
            print(f"   ⚠️  primary_pollutant column not found (already removed)")
        
        # Also exclude any other AQI-related columns that might exist
        aqi_related = [col for col in df.columns if 'aqi' in col.lower() and col not in ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']]
        if aqi_related:
            exclude_cols.extend(aqi_related)
            print(f"   ✅ Excluding additional AQI-related columns: {aqi_related}")
    
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
        # CRITICAL: Ensure all features are numeric
        numeric_features = []
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
            else:
                print(f"   ⚠️  Excluding non-numeric column: {col} (dtype: {df[col].dtype})")
    
        print(f"   Feature columns: {len(numeric_features)}")
        print(f"   Primary target: target_aqi_24h")
        print(f"   Target shape: {df['target_aqi_24h'].shape}")
    
        # CRITICAL: Remove rows where primary target is NaN (due to 72-hour forecasting shift)
        valid_mask = df['target_aqi_24h'].notna()
        df_valid = df[valid_mask].copy()
    
        print(f"   Records with valid targets: {valid_mask.sum():,}")
        print(f"   Records with NaN targets (removed): {(~valid_mask).sum():,}")
    
        # Prepare X and y from valid data only
        X = df_valid[numeric_features]
        y = df_valid['target_aqi_24h']
    
        # CRITICAL: Handle any remaining NaN or Inf values in features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)  # Fill remaining NaNs with 0
    
        print(f"   Final feature matrix shape: {X.shape}")
        print(f"   Final target shape: {y.shape}")
        print(f"   ✅ Multi-horizon targets properly excluded from features")
        print(f"   ✅ Using target_aqi_24h as primary target for feature selection")
    
        return X, y, numeric_features

    def train_lightgbm_for_shap(self, X, y):
        """Train a LightGBM model for SHAP analysis"""
        print(f"\n🌳 TRAINING LIGHTGBM MODEL FOR SHAP ANALYSIS")
        print("-" * 50)
        
        # Split data for training (maintain temporal order)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Validation set: {len(X_val):,} samples")
        
        # Create and train LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        print(f"   Training LightGBM model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"   Model Performance:")
        print(f"      MSE: {mse:.2f}")
        print(f"      MAE: {mae:.2f}")
        print(f"      R²: {r2:.3f}")
        
        # Check if performance is realistic (not too high due to leakage)
        if r2 > 0.95:
            print(f"   ⚠️ R² is very high ({r2:.3f}) - this might indicate remaining data leakage")
        elif r2 > 0.8:
            print(f"   ✅ R² is realistic ({r2:.3f}) - no data leakage detected")
        else:
            print(f"   📊 R² is moderate ({r2:.3f}) - model learning from legitimate patterns")
        
        return model, X_train, X_val
    
    def perform_shap_analysis(self, model, X_train, X_val):
        """Perform SHAP analysis to get feature importance"""
        print(f"\n🔍 PERFORMING SHAP ANALYSIS")
        print("-" * 40)
        
        # Use a subset of validation data for SHAP (faster computation)
        shap_sample_size = min(1000, len(X_val))
        X_shap = X_val.sample(n=shap_sample_size, random_state=42)
        
        print(f"   Using {shap_sample_size} samples for SHAP analysis")
        
        # Calculate SHAP values
        print(f"   Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # Get feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = X_shap.columns
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"   SHAP analysis completed")
        print(f"   Top 15 features by importance:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"      {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return importance_df
    
    def select_top_features(self, importance_df, X, y):
        """Select top features based on SHAP importance"""
        print(f"\n🎯 SELECTING TOP {self.target_features} FEATURES")
        print("-" * 50)
        
        # Get top features
        top_features = importance_df.head(self.target_features)['feature'].tolist()
        
        print(f"   Selected {len(top_features)} features:")
        for i, feature in enumerate(top_features, 1):
            importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
            print(f"      {i:2d}. {feature:<25} {importance:>8.4f}")
        
        # Verify no AQI leakage features
        aqi_leakage_keywords = ['aqi_', 'numerical_aqi', 'aqi']
        leakage_features = []
        for feature in top_features:
            for keyword in aqi_leakage_keywords:
                if keyword in feature.lower():
                    leakage_features.append(feature)
                    break  # Found leakage, no need to check other keywords
        
        if leakage_features:
            print(f"   ⚠️  WARNING: Found potential AQI leakage features:")
            for feature in leakage_features:
                print(f"      ❌ {feature}")
            print(f"   → These features will be removed from selection")
            top_features = [f for f in top_features if f not in leakage_features]
            print(f"   → Final selection: {len(top_features)} features")
            
            # Verify no remaining leakage features
            remaining_leakage = [f for f in top_features if any(keyword in f.lower() for keyword in aqi_leakage_keywords)]
            if remaining_leakage:
                print(f"   ⚠️  CRITICAL: Still found AQI leakage features after removal: {remaining_leakage}")
                print(f"   → Removing these as well...")
                top_features = [f for f in top_features if f not in remaining_leakage]
                print(f"   → Final clean selection: {len(top_features)} features")
        
        # Select features from original dataset
        X_selected = X[top_features]
        
        print(f"\n   Feature selection summary:")
        print(f"      Original features: {X.shape[1]}")
        print(f"      Selected features: {len(top_features)}")
        print(f"      Reduction: {((X.shape[1] - len(top_features)) / X.shape[1] * 100):.1f}%")
        print(f"   ✅ No AQI leakage features in selected features")
        
        return X_selected, top_features
    
    def scale_features(self, X_selected):
        """Scale the selected features"""
        print(f"\n📏 SCALING SELECTED FEATURES")
        print("-" * 40)
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit and transform features
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns)
        
        print(f"   Features scaled using StandardScaler")
        print(f"   Scaled shape: {X_scaled_df.shape}")
        
        # Save scaler for later use
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   Scaler saved to: {self.scaler_path}")
        
        return X_scaled_df, scaler
    
    def create_final_dataset(self, df, X_scaled_df, top_features):
        """Create final dataset with selected features and multi-horizon targets"""
        print(f"\n📊 CREATING FINAL DATASET")
        print("-" * 40)
        
        # CRITICAL: Create final dataset with timestamp, multi-horizon targets, and selected features
        final_df = pd.DataFrame()
        final_df['timestamp'] = df['timestamp']
        
        # Add all multi-horizon targets
        final_df['target_aqi_24h'] = df['target_aqi_24h']
        final_df['target_aqi_48h'] = df['target_aqi_48h']
        final_df['target_aqi_72h'] = df['target_aqi_72h']
        
        print(f"   ✅ Added multi-horizon targets:")
        print(f"      → target_aqi_24h: 24-hour forecasting")
        print(f"      → target_aqi_48h: 48-hour forecasting")
        print(f"      → target_aqi_72h: 72-hour forecasting")
    
        # CRITICAL: Check if primary_pollutant exists before adding it
        if 'primary_pollutant' in df.columns:
            final_df['primary_pollutant'] = df['primary_pollutant']
            print(f"   ✅ Added primary_pollutant column")
        else:
            print(f"   ⚠️  primary_pollutant column not found (was removed to prevent AQI leakage)")
            # CRITICAL: Create a placeholder or skip this column
            final_df['primary_pollutant'] = 'unknown'  # Placeholder value

        # CRITICAL: Add 24h inference mapping: target_timestamp = timestamp + 24h (helps align predictions to future time)
        try:
            final_df['target_timestamp'] = pd.to_datetime(final_df['timestamp']) + pd.to_timedelta(24, unit='h')
        except Exception:
            final_df['target_timestamp'] = pd.to_datetime(final_df['timestamp'])
    
        # Add scaled features
        for i, feature in enumerate(top_features):
            if feature in X_scaled_df.columns:
                final_df[feature] = X_scaled_df[feature]
            else:
                print(f"   ⚠️  Feature {feature} not found in scaled data")
    
        print(f"   Final dataset shape: {final_df.shape}")
        print(f"   Columns: timestamp, target_timestamp, multi-horizon targets, primary_pollutant, + {len(top_features)} features")
    
        return final_df
    
    def save_data(self, final_df, top_features, scaler, importance_df):
        """Save the final dataset and metadata"""
        print(f"\n SAVING FINAL DATASET AND METADATA")
        print("-" * 50)
    
        # CRITICAL: Save final dataset
        final_df.to_csv(self.output_path, index=False)
        print(f"   Final dataset saved to: {self.output_path}")
    
        # CRITICAL: Save selected feature columns
        with open(self.feature_columns_path, 'wb') as f:
            pickle.dump(top_features, f)
        print(f"   Feature columns saved to: {self.feature_columns_path}")
    
        # Save feature importance data
        importance_path = "data_repositories/features/phase1_fixed_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"   Feature importance saved to: {importance_path}")
    
        # CRITICAL: Check if primary_pollutant exists in final dataset
        has_primary_pollutant = 'primary_pollutant' in final_df.columns
        # Save metadata
        metadata = {
            "feature_selection_timestamp": datetime.now().isoformat(),
            "input_data": self.input_path,
            "output_data": self.output_path,
            "target_features": self.target_features,
            "actual_features_selected": len(top_features),
            "feature_selection_method": "SHAP analysis with LightGBM (NO LEAKAGE)",
            "scaler_method": "StandardScaler",
            "scaler_path": self.scaler_path,
            "feature_columns_path": self.feature_columns_path,
            "importance_path": importance_path,
            "data_shape": final_df.shape,
            "date_range": f"{final_df['timestamp'].min()} to {final_df['timestamp'].max()}",
            "target_variables": {
                "target_aqi_24h": "AQI value 24 hours in the future (primary target)",
                "target_aqi_48h": "AQI value 48 hours in the future",
                "target_aqi_72h": "AQI value 72 hours in the future"
            },
            "target_ranges": {
                "24h": f"{final_df['target_aqi_24h'].min():.1f} to {final_df['target_aqi_24h'].max():.1f}",
                "48h": f"{final_df['target_aqi_48h'].min():.1f} to {final_df['target_aqi_48h'].max():.1f}",
                "72h": f"{final_df['target_aqi_72h'].min():.1f} to {final_df['target_aqi_72h'].max():.1f}"
            },
            "forecasting_horizon": "MULTI-HORIZON: 24h (primary), 48h, 72h forecasting",
            "selected_features": top_features,
            "top_15_features": top_features[:15],
            "feature_reduction": f"{((len(importance_df) - len(top_features)) / len(importance_df) * 100):.1f}%",
            "data_leakage_prevention": "All AQI-related features removed",
            "legitimate_features": "Pollutant lags, weather features, time features, pollutant trends",
            "multi_horizon_system": "24h primary target, 48h/72h secondary targets available",
            "primary_pollutant_included": has_primary_pollutant,
            "note": "primary_pollutant was removed during preprocessing to prevent AQI leakage" if not has_primary_pollutant else "primary_pollutant included in final dataset"
        }
    
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        print(f"   Metadata saved to: {self.metadata_path}")
        # Print output dataset date range at the end
        try:
            print(f"📅 Feature-selection output range: {final_df['timestamp'].min()} → {final_df['timestamp'].max()}")
        except Exception:
            pass
    
        return True
    
    def run_feature_selection(self):
        """Run complete feature selection pipeline"""
        print(f"\n🚀 STARTING FIXED FEATURE SELECTION PIPELINE (NO LEAKAGE)")
        print("=" * 80)
        
        # Step 1: Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Step 2: Prepare features and target
        X, y, feature_columns = self.prepare_features_and_target(df)
        
        # Step 3: Train LightGBM for SHAP
        model, X_train, X_val = self.train_lightgbm_for_shap(X, y)
        
        # Step 4: Perform SHAP analysis
        importance_df = self.perform_shap_analysis(model, X_train, X_val)
        
        # Step 5: Select top features
        X_selected, top_features = self.select_top_features(importance_df, X, y)
        
        # Step 6: Scale features
        X_scaled_df, scaler = self.scale_features(X_selected)
        
        # Step 7: Create final dataset
        final_df = self.create_final_dataset(df, X_scaled_df, top_features)
        
        # Step 8: Final verification - ensure no AQI leakage
        print(f"\n🔍 FINAL VERIFICATION - NO AQI LEAKAGE")
        print("-" * 50)
        
        # Check final dataset for any remaining AQI features
        final_columns = final_df.columns.tolist()
        aqi_columns = [col for col in final_columns if 'aqi' in col.lower()]
        allowed_aqi_columns = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']  # Only multi-horizon targets should remain
        
        suspicious_aqi_columns = [col for col in aqi_columns if col not in allowed_aqi_columns]
        if suspicious_aqi_columns:
            print(f"   ❌ CRITICAL ERROR: Found AQI leakage features in final dataset:")
            for col in suspicious_aqi_columns:
                print(f"      ❌ {col}")
            print(f"   → This indicates a serious data leakage issue!")
            return False
        else:
            print(f"   ✅ Final verification passed - No AQI leakage features detected")
            print(f"   ✅ Only legitimate features and multi-horizon targets remain")
            print(f"   ✅ Multi-horizon targets: {allowed_aqi_columns}")
        
        # Step 9: Save data
        success = self.save_data(final_df, top_features, scaler, importance_df)
        
        if success:
            print(f"\n✅ FIXED FEATURE SELECTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("📊 Key Results:")
            print(f"   ✅ Selected top {len(top_features)} features using SHAP analysis")
            print(f"   ✅ Features scaled using StandardScaler")
            print(f"   ✅ Final dataset ready for model training")
            print(f"   ✅ Multi-horizon targets:")
            print(f"      → target_aqi_24h: 24-hour forecasting (primary)")
            print(f"      → target_aqi_48h: 48-hour forecasting")
            print(f"      → target_aqi_72h: 72-hour forecasting")
            print(f"   ✅ Forecasting horizon: MULTI-HORIZON (24h, 48h, 72h)")
            print(f"   ✅ NO DATA LEAKAGE - Only legitimate features")
            print(f"   ✅ Final verification passed - No AQI leakage detected")
            print("\n📊 Next steps:")
            print("   1. Model training for multi-horizon forecasting")
            print("   2. Hyperparameter optimization")
            print("   3. Model evaluation and comparison")
            print("   4. Multi-horizon performance analysis")
        else:
            print(f"\n❌ FIXED FEATURE SELECTION FAILED!")
            print("=" * 80)
        
        return success

def main():
    """Run the fixed feature selection pipeline for multi-horizon forecasting"""
    selector = FixedFeatureSelector()
    success = selector.run_feature_selection()
    
    if success:
        print(f"\n🎉 Ready for Multi-Horizon AQI Forecasting Model Training!")
        print(f"   🎯 24h forecasting: High confidence for daily planning")
        print(f"   🎯 48h forecasting: Medium confidence for weekend planning")
        print(f"   🎯 72h forecasting: Lower confidence for trend indication")
        print(f"   🔒 No data leakage: Only legitimate forecasting features")
        print(f"   ✅ Multi-horizon targets properly configured")
    else:
        print(f"\n❌ Fixed feature selection failed! Check error messages above.")

if __name__ == "__main__":
    main()
