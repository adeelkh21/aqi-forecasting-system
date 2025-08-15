"""
Model Loader for Streamlit App
==============================

This module loads pre-trained models and feature engineering components for real-time inference.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. LSTM model will not be loaded.")

class ModelLoader:
    def __init__(self, models_dir="saved_models", features_dir="saved_features"):
        """Initialize the model loader"""
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        self.metadata_file = self.models_dir / "model_metadata.json"
        
        # Model storage
        self.models = {}
        self.feature_components = {}
        self.metadata = {}
        
        # Status
        self.models_loaded = False
        self.loading_errors = []
        
        # Load everything
        self._load_all_components()
    
    def _load_all_components(self):
        """Load all models and feature engineering components"""
        print("🔍 Loading Pre-trained Models and Components...")
        
        try:
            # Load metadata
            if self._load_metadata():
                # Load feature engineering components
                self._load_feature_components()
                
                # Load ML models
                self._load_ml_models()
                
                # Verify loading
                self._verify_loading()
                
            else:
                raise Exception("Failed to load metadata")
                
        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.loading_errors.append(error_msg)
    
    def _load_metadata(self):
        """Load training metadata"""
        try:
            if not self.metadata_file.exists():
                raise Exception(f"Metadata file not found: {self.metadata_file}")
            
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            print(f"✅ Metadata loaded: {len(self.metadata.get('models', {}))} models")
            return True
            
        except Exception as e:
            print(f"❌ Metadata loading failed: {str(e)}")
            return False
    
    def _load_feature_components(self):
        """Load feature engineering components"""
        print("🔧 Loading Feature Engineering Components...")
        
        components_to_load = [
            ("scaler", "feature_scaler.pkl"),
            ("lstm_scaler_X", "lstm_scaler_X.pkl"),
            ("lstm_scaler_y", "lstm_scaler_y.pkl"),
            ("feature_columns", "feature_columns.pkl"),
            ("lstm_feature_columns", "lstm_feature_columns.pkl")
        ]
        
        for component_name, filename in components_to_load:
            try:
                file_path = self.features_dir / filename
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        self.feature_components[component_name] = pickle.load(f)
                    print(f"   ✅ {component_name} loaded")
                else:
                    print(f"   ⚠️ {component_name} not found: {file_path}")
                    
            except Exception as e:
                print(f"   ❌ Failed to load {component_name}: {str(e)}")
                self.loading_errors.append(f"Failed to load {component_name}: {str(e)}")
    
    def _load_ml_models(self):
        """Load all ML models"""
        print("🤖 Loading ML Models...")
        
        models_to_load = [
            ("random_forest", "random_forest_model.pkl"),
            ("gradient_boosting", "gradient_boosting_model.pkl"),
            ("prophet", "prophet_model.pkl"),
            ("sarima", "sarima_model.pkl")
        ]
        
        # Load pickle-based models
        for model_name, filename in models_to_load:
            try:
                file_path = self.models_dir / filename
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"   ✅ {model_name} loaded")
                else:
                    print(f"   ⚠️ {model_name} not found: {file_path}")
                    
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {str(e)}")
                self.loading_errors.append(f"Failed to load {model_name}: {str(e)}")
        
        # Load LSTM model (TensorFlow format)
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_path = self.models_dir / "lstm_model.keras"
                if lstm_path.exists():
                    self.models["lstm"] = keras.models.load_model(str(lstm_path))
                    print(f"   ✅ LSTM model loaded")
                else:
                    print(f"   ⚠️ LSTM model not found: {lstm_path}")
                    
            except Exception as e:
                print(f"   ❌ Failed to load LSTM: {str(e)}")
                self.loading_errors.append(f"Failed to load LSTM: {str(e)}")
        else:
            print("   ⚠️ LSTM model skipped (TensorFlow not available)")
    
    def _verify_loading(self):
        """Verify that all components loaded successfully"""
        print("\n🔍 Verifying Model Loading...")
        
        # Check models
        models_loaded = len(self.models)
        expected_models = len(self.metadata.get('models', {}))
        
        print(f"   Models: {models_loaded}/{expected_models} loaded")
        
        # Check feature components
        components_loaded = len(self.feature_components)
        expected_components = len(self.metadata.get('feature_engineering', {}))
        
        print(f"   Feature Components: {components_loaded}/{expected_components} loaded")
        
        # Check for critical errors
        if self.loading_errors:
            print(f"   ⚠️ {len(self.loading_errors)} loading errors encountered")
            for error in self.loading_errors:
                print(f"      - {error}")
        
        # Determine overall status
        if models_loaded > 0 and components_loaded > 0:
            self.models_loaded = True
            print("✅ Model loading completed successfully!")
        else:
            print("❌ Model loading failed - insufficient components loaded")
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        return self.models.get(model_name)
    
    def get_feature_component(self, component_name):
        """Get a specific feature component by name"""
        return self.feature_components.get(component_name)
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_performance(self):
        """Get model performance metrics from metadata"""
        return self.metadata.get('performance_metrics', {})
    
    def get_training_info(self):
        """Get training session information"""
        return {
            "training_timestamp": self.metadata.get('training_timestamp'),
            "data_info": self.metadata.get('data_info', {}),
            "models_count": len(self.models),
            "components_count": len(self.feature_components)
        }
    
    def is_ready(self):
        """Check if the model loader is ready for inference"""
        return self.models_loaded and len(self.models) > 0
    
    def get_status_summary(self):
        """Get a summary of the loading status"""
        return {
            "models_loaded": self.models_loaded,
            "models_count": len(self.models),
            "components_count": len(self.feature_components),
            "errors_count": len(self.loading_errors),
            "available_models": self.get_available_models(),
            "training_info": self.get_training_info()
        }

def test_model_loader():
    """Test the model loader"""
    print("🧪 Testing Model Loader...")
    
    loader = ModelLoader()
    
    if loader.is_ready():
        print("✅ Model loader is ready!")
        
        # Print status
        status = loader.get_status_summary()
        print(f"📊 Status: {status['models_count']} models, {status['components_count']} components")
        print(f"🤖 Available models: {', '.join(status['available_models'])}")
        
        # Print training info
        training_info = status['training_info']
        if training_info.get('training_timestamp'):
            print(f"📅 Last trained: {training_info['training_timestamp']}")
        
    else:
        print("❌ Model loader is not ready!")
        print("💡 Make sure to run train_and_save_models.py first")

if __name__ == "__main__":
    test_model_loader()
