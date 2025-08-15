"""
Model Training and Persistence Script
====================================

This script trains all ML models and saves them to disk for later use in the Streamlit app.
"""

import os
import pickle
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import the forecasting system
from enhanced_aqi_forecasting_real import EnhancedAQIForecaster

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer"""
        self.models_dir = "saved_models"
        self.features_dir = "saved_features"
        self.metadata_file = "model_metadata.json"
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Initialize forecaster
        print("🚀 Initializing Enhanced AQI Forecasting System...")
        self.forecaster = EnhancedAQIForecaster()
        
        # Store metadata
        self.metadata = {
            "training_timestamp": datetime.now().isoformat(),
            "models": {},
            "feature_engineering": {},
            "performance_metrics": {},
            "data_info": {}
        }
    
    def train_all_models(self):
        """Train all models and save them"""
        print("\n🎯 Starting Model Training Phase...")
        print("=" * 50)
        
        try:
            # Step 1: Train models using existing system
            print("📊 Training models with historical data...")
            location_data = {
                "latitude": 34.0083,
                "longitude": 71.5189,
                "city": "Peshawar",
                "country": "Pakistan"
            }
            self.forecaster.run_enhanced_forecasting(location_data)
            
            # Step 2: Extract and save individual models
            self._extract_and_save_models()
            
            # Step 3: Save feature engineering pipeline
            self._save_feature_engineering_pipeline()
            
            # Step 4: Save metadata
            self._save_metadata()
            
            print("\n✅ All models trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return False
    
    def _extract_and_save_models(self):
        """Extract trained models and save them individually"""
        print("\n💾 Saving Individual Models...")
        
        # Check if models exist
        if not hasattr(self.forecaster, 'models') or not self.forecaster.models:
            print("   ⚠️ No models found in forecaster.models")
            return
        
        # Save Random Forest
        if 'random_forest' in self.forecaster.models:
            rf_model = self.forecaster.models['random_forest']['model']
            rf_path = os.path.join(self.models_dir, "random_forest_model.pkl")
            with open(rf_path, 'wb') as f:
                pickle.dump(rf_model, f)
            print(f"   ✅ Random Forest saved: {rf_path}")
            
            # Store metadata
            self.metadata["models"]["random_forest"] = {
                "path": rf_path,
                "type": "RandomForestRegressor",
                "saved": True
            }
        
        # Save Gradient Boosting
        if 'gradient_boosting' in self.forecaster.models:
            gb_model = self.forecaster.models['gradient_boosting']['model']
            gb_path = os.path.join(self.models_dir, "gradient_boosting_model.pkl")
            with open(gb_path, 'wb') as f:
                pickle.dump(gb_model, f)
            print(f"   ✅ Gradient Boosting saved: {gb_path}")
            
            self.metadata["models"]["gradient_boosting"] = {
                "path": gb_path,
                "type": "GradientBoostingRegressor",
                "saved": True
            }
        
        # Save LSTM model
        if 'lstm' in self.forecaster.models:
            lstm_model = self.forecaster.models['lstm']['model']
            lstm_path = os.path.join(self.models_dir, "lstm_model.keras")
            lstm_model.save(lstm_path)
            print(f"   ✅ LSTM model saved: {lstm_path}")
            
            self.metadata["models"]["lstm"] = {
                "path": lstm_path,
                "type": "LSTM",
                "saved": True
            }
        
        # Save Prophet model
        if 'prophet' in self.forecaster.models:
            prophet_model = self.forecaster.models['prophet']['model']
            prophet_path = os.path.join(self.models_dir, "prophet_model.pkl")
            with open(prophet_path, 'wb') as f:
                pickle.dump(prophet_model, f)
            print(f"   ✅ Prophet model saved: {prophet_path}")
            
            self.metadata["models"]["prophet"] = {
                "path": prophet_path,
                "type": "Prophet",
                "saved": True
            }
        
        # Save SARIMA model
        if 'sarima' in self.forecaster.models:
            sarima_model = self.forecaster.models['sarima']['model']
            sarima_path = os.path.join(self.models_dir, "sarima_model.pkl")
            with open(sarima_path, 'wb') as f:
                pickle.dump(sarima_model, f)
            print(f"   ✅ SARIMA model saved: {sarima_path}")
            
            self.metadata["models"]["sarima"] = {
                "path": sarima_path,
                "type": "SARIMAX",
                "saved": True
            }
    
    def _save_feature_engineering_pipeline(self):
        """Save feature engineering components"""
        print("\n🔧 Saving Feature Engineering Pipeline...")
        
        # Save feature scaler
        if hasattr(self.forecaster, 'scaler') and self.forecaster.scaler:
            scaler_path = os.path.join(self.features_dir, "feature_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.forecaster.scaler, f)
            print(f"   ✅ Feature scaler saved: {scaler_path}")
            
            self.metadata["feature_engineering"]["scaler"] = {
                "path": scaler_path,
                "type": "StandardScaler",
                "saved": True
            }
        
        # Save LSTM-specific scalers
        if hasattr(self.forecaster, 'lstm_scaler_X') and self.forecaster.lstm_scaler_X:
            lstm_scaler_X_path = os.path.join(self.features_dir, "lstm_scaler_X.pkl")
            with open(lstm_scaler_X_path, 'wb') as f:
                pickle.dump(self.forecaster.lstm_scaler_X, f)
            print(f"   ✅ LSTM feature scaler saved: {lstm_scaler_X_path}")
            
            self.metadata["feature_engineering"]["lstm_scaler_X"] = {
                "path": lstm_scaler_X_path,
                "type": "StandardScaler",
                "saved": True
            }
        
        if hasattr(self.forecaster, 'lstm_scaler_y') and self.forecaster.lstm_scaler_y:
            lstm_scaler_y_path = os.path.join(self.features_dir, "lstm_scaler_y.pkl")
            with open(lstm_scaler_y_path, 'wb') as f:
                pickle.dump(self.forecaster.lstm_scaler_y, f)
            print(f"   ✅ LSTM target scaler saved: {lstm_scaler_y_path}")
            
            self.metadata["feature_engineering"]["lstm_scaler_y"] = {
                "path": lstm_scaler_y_path,
                "type": "MinMaxScaler",
                "saved": True
            }
        
        # Save feature columns
        if hasattr(self.forecaster, 'feature_columns') and self.forecaster.feature_columns:
            feature_cols_path = os.path.join(self.features_dir, "feature_columns.pkl")
            with open(feature_cols_path, 'wb') as f:
                pickle.dump(self.forecaster.feature_columns, f)
            print(f"   ✅ Feature columns saved: {feature_cols_path}")
            
            self.metadata["feature_engineering"]["feature_columns"] = {
                "path": feature_cols_path,
                "count": len(self.forecaster.feature_columns),
                "saved": True
            }
        
        # Save LSTM feature selection info
        if hasattr(self.forecaster, 'lstm_feature_columns') and self.forecaster.lstm_feature_columns:
            lstm_feature_cols_path = os.path.join(self.features_dir, "lstm_feature_columns.pkl")
            with open(lstm_feature_cols_path, 'wb') as f:
                pickle.dump(self.forecaster.lstm_feature_columns, f)
            print(f"   ✅ LSTM feature columns saved: {lstm_feature_cols_path}")
            
            self.metadata["feature_engineering"]["lstm_feature_columns"] = {
                "path": lstm_feature_cols_path,
                "count": len(self.forecaster.lstm_feature_columns),
                "saved": True
            }
        else:
            # Try to get from models if available
            if hasattr(self.forecaster, 'models') and 'lstm' in self.forecaster.models:
                lstm_info = self.forecaster.models['lstm']
                if 'feature_columns' in lstm_info:
                    lstm_feature_cols_path = os.path.join(self.features_dir, "lstm_feature_columns.pkl")
                    with open(lstm_feature_cols_path, 'wb') as f:
                        pickle.dump(lstm_info['feature_columns'], f)
                    print(f"   ✅ LSTM feature columns saved: {lstm_feature_cols_path}")
                    
                    self.metadata["feature_engineering"]["lstm_feature_columns"] = {
                        "path": lstm_feature_cols_path,
                        "count": len(lstm_info['feature_columns']),
                        "saved": True
                    }
    
    def _save_metadata(self):
        """Save comprehensive metadata about the training session"""
        print("\n📋 Saving Training Metadata...")
        
        # Add performance metrics if available
        if hasattr(self.forecaster, 'models') and self.forecaster.models:
            self.metadata["performance_metrics"] = self.forecaster.models
        
        # Add data information
        if hasattr(self.forecaster, 'historical_data') and self.forecaster.historical_data is not None:
            self.metadata["data_info"] = {
                "shape": self.forecaster.historical_data.shape,
                "date_range": {
                    "start": str(self.forecaster.historical_data['date'].min()),
                    "end": str(self.forecaster.historical_data['date'].max())
                },
                "total_records": len(self.forecaster.historical_data)
            }
        
        # Save metadata
        metadata_path = os.path.join(self.models_dir, self.metadata_file)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"   ✅ Metadata saved: {metadata_path}")
        
        # Print summary
        self._print_training_summary()
    
    def _print_training_summary(self):
        """Print a summary of the training session"""
        print("\n" + "=" * 50)
        print("🎉 TRAINING SESSION SUMMARY")
        print("=" * 50)
        
        print(f"📅 Training Timestamp: {self.metadata['training_timestamp']}")
        print(f"🤖 Models Trained: {len(self.metadata['models'])}")
        print(f"🔧 Feature Components: {len(self.metadata['feature_engineering'])}")
        
        if self.metadata['data_info']:
            print(f"📊 Training Data: {self.metadata['data_info']['total_records']} records")
            print(f"📈 Data Shape: {self.metadata['data_info']['shape']}")
        
        print(f"\n💾 Models saved to: {os.path.abspath(self.models_dir)}")
        print(f"🔧 Features saved to: {os.path.abspath(self.features_dir)}")
        print(f"📋 Metadata saved to: {os.path.join(self.models_dir, self.metadata_file)}")
        
        print("\n✅ Ready for Streamlit app deployment!")

def main():
    """Main training function"""
    print("🌤️ AQI Model Training and Persistence System")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train and save models
    success = trainer.train_all_models()
    
    if success:
        print("\n🎯 Phase 1 Complete: Models trained and saved successfully!")
        print("💡 Next step: Implement Streamlit app with model loading")
    else:
        print("\n❌ Phase 1 Failed: Check error messages above")

if __name__ == "__main__":
    main()
