"""
AQI Prediction System - Feature Engineering Pipeline
================================================

This script processes the collected data and creates engineered features for AQI prediction:
- Processes raw weather and pollution data
- Creates lag features (24-hour for pollutants and weather)
- Performs feature selection using SHAP analysis
- Removes highly correlated features
- Documents feature importance

Author: Data Science Team
Date: 2024-03-09
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineering pipeline"""
        print("🔧 Initializing Feature Engineering Pipeline")
        print("=" * 50)
        
        # Setup directories
        self.data_dir = os.path.join("data_repositories", "merged_data")
        self.feature_dir = os.path.join("data_repositories", "features")
        self.output_dir = os.path.join("data_repositories", "feature_analysis")
        
        # Create directories
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize feature sets as per roadmap
        self.weather_features = ['temperature', 'relative_humidity', 'wind_speed', 'pressure']
        self.pollution_features = ['pm2_5', 'pm10', 'no2', 'o3']
        self.time_features = ['hour', 'day_of_week', 'is_weekend']
        
        print(f"📂 Input data: {self.data_dir}")
        print(f"📁 Features output: {self.feature_dir}")
        print(f"📊 Analysis output: {self.output_dir}")

    def load_data(self):
        """Load merged dataset"""
        print("\n📊 Loading Merged Dataset")
        print("-" * 30)
        
        try:
            data_file = os.path.join(self.data_dir, "processed", "merged_data.csv")
            
            if not os.path.exists(data_file):
                print(f"❌ Data file not found: {data_file}")
                return None
            
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            print(f"✅ Data loaded successfully")
            print(f"📊 Shape: {df.shape}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"🎯 Target variable: aqi_numeric")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    def create_core_features(self, df):
        """Create core feature set from roadmap requirements"""
        print("\n🏗️ Creating Core Features")
        print("-" * 30)
        
        try:
            df_features = df.copy()
            
            # Weather Features (as per roadmap)
            print("🌤️ Processing weather features...")
            weather_cols = []
            for feature in self.weather_features:
                if feature in df_features.columns:
                    weather_cols.append(feature)
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ❌ {feature} - not found")
            
            # Pollution Features (as per roadmap)
            print("🏭 Processing pollution features...")
            pollution_cols = []
            for feature in self.pollution_features:
                if feature in df_features.columns:
                    pollution_cols.append(feature)
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ❌ {feature} - not found")
            
            # Time Features (as per roadmap)
            print("⏰ Processing time features...")
            time_cols = []
            for feature in self.time_features:
                if feature in df_features.columns:
                    time_cols.append(feature)
                    print(f"   ✅ {feature}")
                else:
                    print(f"   ❌ {feature} - not found")
            
            # Additional time features if missing
            if 'month' not in df_features.columns:
                df_features['month'] = df_features['timestamp'].dt.month
                time_cols.append('month')
                print(f"   ✅ month (created)")
            
            # Core feature set
            self.core_features = weather_cols + pollution_cols + time_cols
            
            print(f"\n📋 Core Features Summary:")
            print(f"   Weather: {len(weather_cols)} features")
            print(f"   Pollution: {len(pollution_cols)} features") 
            print(f"   Time: {len(time_cols)} features")
            print(f"   Total: {len(self.core_features)} features")
            
            return df_features
            
        except Exception as e:
            print(f"❌ Error creating core features: {str(e)}")
            return None

    def create_lag_features(self, df):
        """Create 24-hour lag features for pollutants and weather"""
        print("\n⏳ Creating Lag Features")
        print("-" * 30)
        
        try:
            df_lag = df.copy()
            lag_features = []
            
            # 24-hour lag for main pollutants (as per roadmap)
            print("🏭 Creating 24h lag features for pollutants...")
            pollutant_lag_cols = ['pm2_5', 'pm10', 'no2', 'o3']
            for col in pollutant_lag_cols:
                if col in df_lag.columns:
                    lag_col = f"{col}_lag24h"
                    df_lag[lag_col] = df_lag[col].shift(24)
                    lag_features.append(lag_col)
                    print(f"   ✅ {lag_col}")
            
            # 24-hour lag for weather (as per roadmap) 
            print("🌤️ Creating 24h lag features for weather...")
            weather_lag_cols = ['temperature', 'relative_humidity', 'wind_speed', 'pressure']
            for col in weather_lag_cols:
                if col in df_lag.columns:
                    lag_col = f"{col}_lag24h"
                    df_lag[lag_col] = df_lag[col].shift(24)
                    lag_features.append(lag_col)
                    print(f"   ✅ {lag_col}")
            
            # Additional useful lag features
            print("📊 Creating additional lag features...")
            additional_lags = [
                ('aqi_numeric', 'aqi_lag1h', 1),
                ('aqi_numeric', 'aqi_lag3h', 3),
                ('aqi_numeric', 'aqi_lag6h', 6),
                ('pm2_5', 'pm25_lag6h', 6),
                ('pm10', 'pm10_lag6h', 6)
            ]
            
            for base_col, lag_col, shift_hours in additional_lags:
                if base_col in df_lag.columns:
                    df_lag[lag_col] = df_lag[base_col].shift(shift_hours)
                    lag_features.append(lag_col)
                    print(f"   ✅ {lag_col}")
            
            # Store lag feature names
            self.lag_features = lag_features
            
            print(f"\n📋 Lag Features Summary:")
            print(f"   Total lag features created: {len(lag_features)}")
            print(f"   24h pollutant lags: {len([f for f in lag_features if 'lag24h' in f and any(p in f for p in pollutant_lag_cols)])}")
            print(f"   24h weather lags: {len([f for f in lag_features if 'lag24h' in f and any(w in f for w in weather_lag_cols)])}")
            
            return df_lag
            
        except Exception as e:
            print(f"❌ Error creating lag features: {str(e)}")
            return None

    def perform_feature_analysis(self, df):
        """Run SHAP analysis to identify key features"""
        print("\n🔍 Performing Feature Analysis with SHAP")
        print("-" * 40)
        
        try:
            # Prepare feature set
            all_features = self.core_features + self.lag_features
            feature_cols = [col for col in all_features if col in df.columns]
            
            # Remove rows with NaN values (due to lag features)
            df_clean = df[['aqi_numeric'] + feature_cols].dropna()
            
            if len(df_clean) < 100:
                print("❌ Insufficient data after removing NaN values")
                return None, None
            
            X = df_clean[feature_cols]
            y = df_clean['aqi_numeric']
            
            print(f"📊 Analysis dataset shape: {X.shape}")
            print(f"📊 Target variable range: {y.min():.1f} - {y.max():.1f}")
            
            # Train a Random Forest model for SHAP analysis
            print("🌳 Training Random Forest for SHAP analysis...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"✅ Model trained. R² score: {rf_model.score(X_test, y_test):.3f}")
            
            # SHAP Analysis
            print("🔍 Running SHAP analysis...")
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test.iloc[:500])  # Sample for performance
            
            # Calculate SHAP feature importance
            shap_importance = pd.DataFrame({
                'feature': X.columns,
                'shap_importance': np.abs(shap_values).mean(0)
            }).sort_values('shap_importance', ascending=False)
            
            # Combine importance metrics
            importance_combined = feature_importance.merge(shap_importance, on='feature')
            importance_combined['combined_score'] = (
                importance_combined['importance'] * 0.5 + 
                importance_combined['shap_importance'] * 0.5
            )
            importance_combined = importance_combined.sort_values('combined_score', ascending=False)
            
            print("✅ SHAP analysis completed")
            print("\n🏆 Top 10 Most Important Features:")
            for i, row in importance_combined.head(10).iterrows():
                print(f"   {row['feature']:.<25} {row['combined_score']:.4f}")
            
            return importance_combined, shap_values
            
        except Exception as e:
            print(f"❌ Error in feature analysis: {str(e)}")
            return None, None

    def remove_correlated_features(self, df, importance_df, correlation_threshold=0.9):
        """Remove highly correlated features"""
        print(f"\n🔗 Removing Highly Correlated Features (threshold: {correlation_threshold})")
        print("-" * 50)
        
        try:
            # Get feature columns
            all_features = self.core_features + self.lag_features
            feature_cols = [col for col in all_features if col in df.columns]
            
            # Calculate correlation matrix
            corr_matrix = df[feature_cols].corr().abs()
            
            # Find highly correlated feature pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to remove
            to_remove = []
            for column in upper_triangle.columns:
                correlated_features = upper_triangle.index[upper_triangle[column] > correlation_threshold].tolist()
                if correlated_features:
                    # For each correlated pair, keep the more important feature
                    for corr_feature in correlated_features:
                        col_importance = importance_df[importance_df['feature'] == column]['combined_score'].iloc[0]
                        corr_importance = importance_df[importance_df['feature'] == corr_feature]['combined_score'].iloc[0]
                        
                        # Remove the less important feature
                        if col_importance > corr_importance and corr_feature not in to_remove:
                            to_remove.append(corr_feature)
                        elif col_importance <= corr_importance and column not in to_remove:
                            to_remove.append(column)
            
            # Remove duplicates
            to_remove = list(set(to_remove))
            
            # Keep important features
            final_features = [f for f in feature_cols if f not in to_remove]
            
            print(f"📊 Correlation analysis results:")
            print(f"   Original features: {len(feature_cols)}")
            print(f"   Removed due to correlation: {len(to_remove)}")
            print(f"   Final feature count: {len(final_features)}")
            
            if to_remove:
                print(f"\n❌ Removed features:")
                for feature in to_remove:
                    print(f"   - {feature}")
            
            self.final_features = final_features
            return final_features
            
        except Exception as e:
            print(f"❌ Error removing correlated features: {str(e)}")
            return feature_cols

    def save_features_and_analysis(self, df, importance_df):
        """Save final features and analysis results"""
        print("\n💾 Saving Features and Analysis Results")
        print("-" * 40)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save feature dataset
            feature_df = df[['timestamp', 'aqi_numeric'] + self.final_features].dropna()
            feature_file = os.path.join(self.feature_dir, "engineered_features.csv")
            feature_df.to_csv(feature_file, index=False)
            
            # Save feature importance
            importance_file = os.path.join(self.output_dir, "feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)
            
            # Save feature metadata
            metadata = {
                "timestamp": timestamp,
                "total_records": len(feature_df),
                "date_range": {
                    "start": feature_df['timestamp'].min().isoformat(),
                    "end": feature_df['timestamp'].max().isoformat()
                },
                "feature_engineering": {
                    "core_features": len(self.core_features),
                    "lag_features": len(self.lag_features),
                    "final_features": len(self.final_features),
                    "removed_features": len(self.core_features) + len(self.lag_features) - len(self.final_features)
                },
                "feature_categories": {
                    "weather": [f for f in self.final_features if any(w in f for w in self.weather_features)],
                    "pollution": [f for f in self.final_features if any(p in f for p in self.pollution_features)],
                    "time": [f for f in self.final_features if any(t in f for t in self.time_features)],
                    "lag": [f for f in self.final_features if 'lag' in f]
                },
                "top_10_features": importance_df.head(10)['feature'].tolist()
            }
            
            metadata_file = os.path.join(self.output_dir, "feature_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Feature engineering completed successfully!")
            print(f"📁 Files saved:")
            print(f"   - {feature_file}")
            print(f"   - {importance_file}")
            print(f"   - {metadata_file}")
            
            print(f"\n📊 Final Dataset Summary:")
            print(f"   Records: {len(feature_df):,}")
            print(f"   Features: {len(self.final_features)}")
            print(f"   Date range: {feature_df['timestamp'].min()} to {feature_df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving features: {str(e)}")
            return False

    def run_pipeline(self):
        """Run complete feature engineering pipeline"""
        print("\n🚀 Starting Feature Engineering Pipeline")
        print("=" * 50)
        
        # Step 1: Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Step 2: Create core features
        df = self.create_core_features(df)
        if df is None:
            return False
        
        # Step 3: Create lag features
        df = self.create_lag_features(df)
        if df is None:
            return False
        
        # Step 4: Feature analysis with SHAP
        importance_df, shap_values = self.perform_feature_analysis(df)
        if importance_df is None:
            return False
        
        # Step 5: Remove correlated features
        final_features = self.remove_correlated_features(df, importance_df)
        
        # Step 6: Save results
        success = self.save_features_and_analysis(df, importance_df)
        
        if success:
            print("\n🎉 Feature Engineering Pipeline Completed Successfully!")
            print("✅ Ready for Feature Store Integration (Phase 3)")
        else:
            print("\n❌ Feature Engineering Pipeline Failed!")
        
        return success

def main():
    """Run feature engineering pipeline"""
    engineer = FeatureEngineer()
    success = engineer.run_pipeline()
    
    if not success:
        print("\n❌ Pipeline failed! Check error messages above.")
        exit(1)

if __name__ == "__main__":
    main()
