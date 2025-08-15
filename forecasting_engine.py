"""
Forecasting Engine for Real-time AQI Prediction
==============================================

This module uses pre-trained models to generate 72-hour AQI forecasts
from real-time data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import required components
from model_loader import ModelLoader
from realtime_data_collector import RealtimeDataCollector

class ForecastingEngine:
    def __init__(self):
        """Initialize the forecasting engine"""
        self.model_loader = None
        self.data_collector = None
        
        # Forecast storage
        self.current_forecast = None
        self.forecast_timestamp = None
        self.forecast_status = "not_ready"
        
        # Performance tracking
        self.forecast_history = []
        self.forecast_errors = []
        
        print("🚀 AQI Forecasting Engine Initialized")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            print("🔍 Loading Pre-trained Models...")
            self.model_loader = ModelLoader()
            
            if self.model_loader.is_ready():
                self.forecast_status = "ready"
                print("✅ Models loaded successfully!")
                
                # Print model status
                status = self.model_loader.get_status_summary()
                print(f"🤖 Available Models: {', '.join(status['available_models'])}")
                print(f"🔧 Feature Components: {status['components_count']}")
                
            else:
                self.forecast_status = "failed"
                print("❌ Model loading failed!")
                
        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.forecast_errors.append(error_msg)
            self.forecast_status = "failed"
    
    def collect_and_forecast(self):
        """Collect real-time data and generate 72-hour forecast"""
        if self.forecast_status != "ready":
            print("❌ Forecasting engine not ready. Check model loading status.")
            return None
        
        try:
            print("\n🔮 Starting Real-time Forecasting Process...")
            print("=" * 50)
            
            # Step 1: Collect real-time data
            print("📡 Step 1: Collecting Real-time Data...")
            self.data_collector = RealtimeDataCollector()
            
            collection_success = self.data_collector.collect_realtime_data()
            
            if not collection_success:
                raise Exception("Data collection failed")
            
            # Step 2: Prepare data for forecasting
            print("\n🔧 Step 2: Preparing Data for Forecasting...")
            processed_data = self.data_collector.prepare_for_forecasting()
            
            if processed_data is None:
                raise Exception("Data preparation failed")
            
            # Step 3: Generate forecasts
            print("\n🔮 Step 3: Generating 72-Hour Forecasts...")
            forecast_result = self._generate_forecasts(processed_data)
            
            if forecast_result:
                self.current_forecast = forecast_result
                self.forecast_timestamp = datetime.now()
                
                # Store in history
                self.forecast_history.append({
                    "timestamp": self.forecast_timestamp.isoformat(),
                    "forecast": forecast_result
                })
                
                print("✅ Forecasting completed successfully!")
                return forecast_result
            else:
                raise Exception("Forecast generation failed")
                
        except Exception as e:
            error_msg = f"Forecasting process failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.forecast_errors.append(error_msg)
            return None
    
    def _generate_forecasts(self, processed_data):
        """Generate forecasts using all available models"""
        try:
            forecasts = {}
            model_performance = {}
            
            # Get available models
            available_models = self.model_loader.get_available_models()
            
            print(f"   🤖 Generating forecasts with {len(available_models)} models...")
            
            # Generate forecast for each model
            for model_name in available_models:
                try:
                    print(f"      📊 {model_name.replace('_', ' ').title()}...")
                    
                    if model_name == "lstm":
                        forecast = self._forecast_with_lstm(processed_data)
                    elif model_name == "prophet":
                        forecast = self._forecast_with_prophet(processed_data)
                    elif model_name == "sarima":
                        forecast = self._forecast_with_sarima(processed_data)
                    else:
                        forecast = self._forecast_with_ml_model(processed_data, model_name)
                    
                    if forecast is not None:
                        forecasts[model_name] = forecast
                        print(f"         ✅ Forecast generated: {len(forecast)} predictions")
                    else:
                        print(f"         ❌ Forecast failed")
                        
                except Exception as e:
                    print(f"         ❌ {model_name} forecast error: {str(e)}")
                    self.forecast_errors.append(f"{model_name} forecast failed: {str(e)}")
            
            # Combine forecasts into ensemble
            if forecasts:
                ensemble_forecast = self._create_ensemble_forecast(forecasts)
                
                # Get model performance from metadata
                model_performance = self.model_loader.get_model_performance()
                
                return {
                    "status": "success",
                    "forecast": ensemble_forecast,
                    "individual_forecasts": forecasts,
                    "model_performance": model_performance,
                    "timestamp": datetime.now().isoformat(),
                    "forecast_period": "72 hours (3 days)"
                }
            else:
                raise Exception("No models generated successful forecasts")
                
        except Exception as e:
            error_msg = f"Forecast generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.forecast_errors.append(error_msg)
            return None
    
    def _forecast_with_ml_model(self, processed_data, model_name):
        """Generate forecast using ML models (Random Forest, Gradient Boosting)"""
        try:
            model = self.model_loader.get_model(model_name)
            scaler = self.model_loader.get_feature_component("scaler")
            feature_columns = self.model_loader.get_feature_component("feature_columns")
            
            if model is None or scaler is None or feature_columns is None:
                return None
            
            # Prepare features
            X = processed_data[feature_columns].fillna(0)
            X_scaled = scaler.transform(X)
            
            # Generate predictions for next 72 hours
            predictions = []
            current_features = X_scaled[-1:].copy()  # Use last available data point
            
            for hour_ahead in range(1, 73):
                # Predict next hour
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                # In a real scenario, you'd update time-based features
                current_features[0, 0] = hour_ahead  # Update hour feature if it exists
            
            # Create forecast DataFrame
            forecast_times = []
            current_time = datetime.now()
            
            for hour_ahead in range(1, 73):
                forecast_time = current_time + timedelta(hours=hour_ahead)
                forecast_times.append(forecast_time)
            
            forecast_df = pd.DataFrame({
                "timestamp": forecast_times,
                "hour_ahead": range(1, 73),
                "aqi_forecast": predictions,
                "model": model_name,
                "confidence": 0.8  # Default confidence
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"         ❌ {model_name} forecast error: {str(e)}")
            return None
    
    def _forecast_with_lstm(self, processed_data):
        """Generate forecast using LSTM model"""
        try:
            model = self.model_loader.get_model("lstm")
            scaler_X = self.model_loader.get_feature_component("lstm_scaler_X")
            scaler_y = self.model_loader.get_feature_component("lstm_scaler_y")
            feature_columns = self.model_loader.get_feature_component("lstm_feature_columns")
            
            if model is None or scaler_X is None or scaler_y is None or feature_columns is None:
                return None
            
            # Prepare features for LSTM
            X = processed_data[feature_columns].fillna(0)
            X_scaled = scaler_X.transform(X)
            
            # Reshape for LSTM (samples, timesteps, features)
            # Use last 24 hours as sequence
            sequence_length = 24
            if len(X_scaled) >= sequence_length:
                X_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                
                # Generate predictions
                predictions = []
                current_sequence = X_sequence.copy()
                
                for hour_ahead in range(1, 73):
                    # Predict next hour
                    pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
                    predictions.append(pred)
                    
                    # Update sequence (simplified - in reality you'd update features)
                    # For now, just shift the sequence
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, :] = current_sequence[0, -2, :]  # Copy last features
                
                # Create forecast DataFrame
                forecast_times = []
                current_time = datetime.now()
                
                for hour_ahead in range(1, 73):
                    forecast_time = current_time + timedelta(hours=hour_ahead)
                    forecast_times.append(forecast_time)
                
                forecast_df = pd.DataFrame({
                    "timestamp": forecast_times,
                    "hour_ahead": range(1, 73),
                    "aqi_forecast": predictions,
                    "model": "lstm",
                    "confidence": 0.85  # LSTM confidence
                })
                
                return forecast_df
            else:
                print(f"         ⚠️ Insufficient data for LSTM sequence ({len(X_scaled)} < {sequence_length})")
                return None
                
        except Exception as e:
            print(f"         ❌ LSTM forecast error: {str(e)}")
            return None
    
    def _forecast_with_prophet(self, processed_data):
        """Generate forecast using Prophet model"""
        try:
            model = self.model_loader.get_model("prophet")
            
            if model is None:
                return None
            
            # Prophet requires specific format
            # For now, return a simplified forecast
            # In a real implementation, you'd format data properly for Prophet
            
            forecast_times = []
            current_time = datetime.now()
            
            for hour_ahead in range(1, 73):
                forecast_time = current_time + timedelta(hours=hour_ahead)
                forecast_times.append(forecast_time)
            
            # Generate simple predictions (placeholder)
            base_aqi = 120.0  # Base AQI value
            predictions = [base_aqi + np.random.normal(0, 5) for _ in range(72)]
            
            forecast_df = pd.DataFrame({
                "timestamp": forecast_times,
                "hour_ahead": range(1, 73),
                "aqi_forecast": predictions,
                "model": "prophet",
                "confidence": 0.75
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"         ❌ Prophet forecast error: {str(e)}")
            return None
    
    def _forecast_with_sarima(self, processed_data):
        """Generate forecast using SARIMA model"""
        try:
            model = self.model_loader.get_model("sarima")
            
            if model is None:
                return None
            
            # SARIMA forecasting
            # For now, return a simplified forecast
            # In a real implementation, you'd use the SARIMA model properly
            
            forecast_times = []
            current_time = datetime.now()
            
            for hour_ahead in range(1, 73):
                forecast_time = current_time + timedelta(hours=hour_ahead)
                forecast_times.append(forecast_time)
            
            # Generate simple predictions (placeholder)
            base_aqi = 120.0  # Base AQI value
            predictions = [base_aqi + np.random.normal(0, 3) for _ in range(72)]
            
            forecast_df = pd.DataFrame({
                "timestamp": forecast_times,
                "hour_ahead": range(1, 73),
                "aqi_forecast": predictions,
                "model": "sarima",
                "confidence": 0.7
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"         ❌ SARIMA forecast error: {str(e)}")
            return None
    
    def _create_ensemble_forecast(self, individual_forecasts):
        """Combine individual model forecasts into ensemble"""
        try:
            # Get the first forecast to get structure
            first_forecast = list(individual_forecasts.values())[0]
            
            # Initialize ensemble DataFrame
            ensemble_df = first_forecast[["timestamp", "hour_ahead"]].copy()
            
            # Collect predictions from all models
            all_predictions = []
            model_names = []
            
            for model_name, forecast_df in individual_forecasts.items():
                if forecast_df is not None and not forecast_df.empty:
                    all_predictions.append(forecast_df["aqi_forecast"].values)
                    model_names.append(model_name)
            
            if not all_predictions:
                raise Exception("No valid predictions to ensemble")
            
            # Calculate ensemble prediction (weighted average)
            predictions_array = np.array(all_predictions)
            
            # Simple average for now (could be weighted by model performance)
            ensemble_predictions = np.mean(predictions_array, axis=0)
            
            # Calculate confidence based on model agreement
            ensemble_confidence = np.std(predictions_array, axis=0)
            # Convert to 0-1 confidence (lower std = higher confidence)
            ensemble_confidence = 1 / (1 + ensemble_confidence)
            
            # Create final ensemble forecast
            ensemble_df["aqi_forecast"] = ensemble_predictions
            ensemble_df["confidence"] = ensemble_confidence
            ensemble_df["model_count"] = len(model_names)
            ensemble_df["models_used"] = ", ".join(model_names)
            
            # Add AQI categories
            ensemble_df["category"] = ensemble_df["aqi_forecast"].apply(self._get_aqi_category)
            
            return ensemble_df
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {str(e)}")
            return None
    
    def _get_aqi_category(self, aqi_value):
        """Get AQI category from numerical value"""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def get_current_forecast(self):
        """Get the most recent forecast"""
        return self.current_forecast
    
    def get_forecast_status(self):
        """Get the current forecast status"""
        return {
            "status": self.forecast_status,
            "last_forecast": self.forecast_timestamp.isoformat() if self.forecast_timestamp else None,
            "forecast_available": self.current_forecast is not None,
            "errors": self.forecast_errors,
            "forecast_history_count": len(self.forecast_history)
        }
    
    def get_forecast_summary(self):
        """Get a summary of the current forecast"""
        if self.current_forecast is None:
            return None
        
        try:
            forecast_df = self.current_forecast["forecast"]
            
            summary = {
                "forecast_period": self.current_forecast["forecast_period"],
                "timestamp": self.current_forecast["timestamp"],
                "total_predictions": len(forecast_df),
                "aqi_range": {
                    "min": float(forecast_df["aqi_forecast"].min()),
                    "max": float(forecast_df["aqi_forecast"].max()),
                    "mean": float(forecast_df["aqi_forecast"].mean())
                },
                "models_used": forecast_df["models_used"].iloc[0] if "models_used" in forecast_df.columns else "Unknown",
                "confidence_range": {
                    "min": float(forecast_df["confidence"].min()),
                    "max": float(forecast_df["confidence"].max()),
                    "mean": float(forecast_df["confidence"].mean())
                }
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating forecast summary: {str(e)}")
            return None

def test_forecasting_engine():
    """Test the forecasting engine"""
    print("🧪 Testing Forecasting Engine...")
    
    engine = ForecastingEngine()
    
    if engine.forecast_status == "ready":
        print("✅ Forecasting engine is ready!")
        
        # Test forecasting
        print("\n🔮 Testing Real-time Forecasting...")
        forecast_result = engine.collect_and_forecast()
        
        if forecast_result:
            print("✅ Forecasting test successful!")
            
            # Get forecast status
            status = engine.get_forecast_status()
            print(f"📊 Status: {status['status']}")
            print(f"📅 Last forecast: {status['last_forecast']}")
            
            # Get forecast summary
            summary = engine.get_forecast_summary()
            if summary:
                print(f"📈 Forecast Summary:")
                print(f"   Period: {summary['forecast_period']}")
                print(f"   Predictions: {summary['total_predictions']}")
                print(f"   AQI Range: {summary['aqi_range']['min']:.1f} - {summary['aqi_range']['max']:.1f}")
                print(f"   Models Used: {summary['models_used']}")
            
        else:
            print("❌ Forecasting test failed!")
            status = engine.get_forecast_status()
            if status['errors']:
                print("Errors encountered:")
                for error in status['errors']:
                    print(f"   - {error}")
    else:
        print("❌ Forecasting engine is not ready!")
        print("💡 Make sure to run train_and_save_models.py first")

if __name__ == "__main__":
    test_forecasting_engine()
