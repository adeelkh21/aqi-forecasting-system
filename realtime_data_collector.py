"""
Real-time Data Collector for AQI Forecasting
===========================================

This module collects real-time weather and pollution data for the past 3 days
and prepares it for feature engineering and forecasting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Import data collection and feature engineering
from phase1_data_collection import DataCollector
from enhanced_aqi_forecasting_real import EnhancedAQIForecaster

class RealtimeDataCollector:
    def __init__(self):
        """Initialize the real-time data collector"""
        self.collection_days = 3  # Collect past 3 days
        self.data_collector = None
        self.forecaster = None
        
        # Data storage
        self.raw_weather_data = None
        self.raw_pollution_data = None
        self.merged_data = None
        self.processed_data = None
        
        # Status
        self.last_collection = None
        self.collection_status = "not_started"
        self.collection_errors = []
        
        print("🚀 Real-time AQI Data Collector Initialized")
        print(f"📅 Collection Period: Past {self.collection_days} days")
    
    def collect_realtime_data(self):
        """Collect real-time data for the past 3 days"""
        print(f"\n📡 Starting Real-time Data Collection...")
        print(f"⏰ Collection Period: Past {self.collection_days} days")
        print("=" * 50)
        
        try:
            self.collection_status = "collecting"
            
            # Step 1: Collect weather data
            print("🌤️ Collecting Weather Data...")
            weather_success = self._collect_weather_data()
            
            # Step 2: Collect pollution data
            print("🏭 Collecting Pollution Data...")
            pollution_success = self._collect_pollution_data()
            
            # Step 3: Merge and process data
            if weather_success and pollution_success:
                print("🔄 Merging and Processing Data...")
                merge_success = self._merge_and_process_data()
                
                if merge_success:
                    self.collection_status = "completed"
                    self.last_collection = datetime.now()
                    print("✅ Real-time data collection completed successfully!")
                    return True
                else:
                    self.collection_status = "failed"
                    print("❌ Data merging and processing failed")
                    return False
            else:
                self.collection_status = "failed"
                print("❌ Data collection failed")
                return False
                
        except Exception as e:
            error_msg = f"Data collection error: {str(e)}"
            print(f"❌ {error_msg}")
            self.collection_errors.append(error_msg)
            self.collection_status = "failed"
            return False
    
    def _collect_weather_data(self):
        """Collect weather data from Meteostat"""
        try:
            # Initialize data collector with 3 days
            self.data_collector = DataCollector()
            
            # Override collection period
            self.data_collector.COLLECTION_DAYS = self.collection_days
            self.data_collector.start_date = datetime.now() - timedelta(days=self.collection_days)
            self.data_collector.end_date = datetime.now()
            
            print(f"   📅 Period: {self.data_collector.start_date.date()} to {self.data_collector.end_date.date()}")
            
            # Fetch weather data
            weather_df = self.data_collector.fetch_weather_data()
            
            if weather_df is not None and not weather_df.empty:
                self.raw_weather_data = weather_df
                print(f"   ✅ Weather data collected: {len(weather_df)} records")
                print(f"   📊 Features: {', '.join(weather_df.columns)}")
                return True
            else:
                print("   ❌ No weather data received")
                return False
                
        except Exception as e:
            error_msg = f"Weather collection failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.collection_errors.append(error_msg)
            return False
    
    def _collect_pollution_data(self):
        """Collect pollution data from OpenWeatherMap"""
        try:
            if self.data_collector is None:
                raise Exception("Data collector not initialized")
            
            # Fetch pollution data
            pollution_df = self.data_collector.fetch_pollution_data()
            
            if pollution_df is not None and not pollution_df.empty:
                self.raw_pollution_data = pollution_df
                print(f"   ✅ Pollution data collected: {len(pollution_df)} records")
                print(f"   📊 Features: {', '.join(pollution_df.columns)}")
                return True
            else:
                print("   ❌ No pollution data received")
                return False
                
        except Exception as e:
            error_msg = f"Pollution collection failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.collection_errors.append(error_msg)
            return False
    
    def _merge_and_process_data(self):
        """Merge weather and pollution data and process for forecasting"""
        try:
            if self.raw_weather_data is None or self.raw_pollution_data is None:
                raise Exception("Raw data not available")
            
            # Merge data using the existing pipeline
            merged_df = self.data_collector.merge_and_process_data(
                self.raw_weather_data, 
                self.raw_pollution_data
            )
            
            if merged_df is not None and not merged_df.empty:
                self.merged_data = merged_df
                print(f"   ✅ Data merged: {len(merged_df)} records")
                print(f"   📊 Final features: {', '.join(merged_df.columns)}")
                
                # Save processed data
                self._save_collected_data()
                
                return True
            else:
                print("   ❌ Data merging failed")
                return False
                
        except Exception as e:
            error_msg = f"Data merging failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            self.collection_errors.append(error_msg)
            return False
    
    def _save_collected_data(self):
        """Save the collected and processed data"""
        try:
            # Create timestamp for this collection
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to hourly data repository
            data_dir = self.data_collector.data_dir
            processed_file = os.path.join(data_dir, "processed", f"realtime_data_{timestamp}.csv")
            
            # Save merged data
            self.merged_data.to_csv(processed_file, index=False)
            
            # Also save as latest data
            latest_file = os.path.join(data_dir, "processed", "latest_realtime_data.csv")
            self.merged_data.to_csv(latest_file, index=False)
            
            print(f"   💾 Data saved: {processed_file}")
            print(f"   💾 Latest data: {latest_file}")
            
            # Update data collector's data file reference
            self.data_collector.latest_data_file = latest_file
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save data: {str(e)}")
    
    def get_collected_data(self):
        """Get the collected and processed data"""
        return self.merged_data
    
    def get_collection_status(self):
        """Get the current collection status"""
        return {
            "status": self.collection_status,
            "last_collection": self.last_collection.isoformat() if self.last_collection else None,
            "collection_days": self.collection_days,
            "errors": self.collection_errors,
            "data_available": self.merged_data is not None,
            "data_shape": self.merged_data.shape if self.merged_data is not None else None
        }
    
    def get_data_summary(self):
        """Get a summary of the collected data"""
        if self.merged_data is None:
            return None
        
        try:
            summary = {
                "total_records": len(self.merged_data),
                "date_range": {
                    "start": str(self.merged_data['timestamp'].min()),
                    "end": str(self.merged_data['timestamp'].max())
                },
                "features": list(self.merged_data.columns),
                "missing_values": self.merged_data.isnull().sum().to_dict(),
                "data_types": self.merged_data.dtypes.to_dict()
            }
            
            # Add AQI summary if available
            if 'aqi' in self.merged_data.columns:
                aqi_data = self.merged_data['aqi'].dropna()
                if len(aqi_data) > 0:
                    summary["aqi_summary"] = {
                        "min": float(aqi_data.min()),
                        "max": float(aqi_data.max()),
                        "mean": float(aqi_data.mean()),
                        "std": float(aqi_data.std())
                    }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating data summary: {str(e)}")
            return None
    
    def prepare_for_forecasting(self):
        """Prepare the collected data for forecasting"""
        if self.merged_data is None:
            print("❌ No data available for forecasting preparation")
            return None
        
        try:
            print("🔧 Preparing Data for Forecasting...")
            
            # Initialize forecaster if not already done
            if self.forecaster is None:
                self.forecaster = EnhancedAQIForecaster()
            
            # Use the forecaster's feature engineering
            processed_data = self.forecaster.engineer_enhanced_features(
                self.merged_data.copy(), 
                is_training=False
            )
            
            if processed_data is not None and not processed_data.empty:
                self.processed_data = processed_data
                print(f"✅ Data prepared for forecasting: {len(processed_data)} records")
                print(f"📊 Features: {len(processed_data.columns)} columns")
                return processed_data
            else:
                print("❌ Feature engineering failed")
                return None
                
        except Exception as e:
            error_msg = f"Forecasting preparation failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.collection_errors.append(error_msg)
            return None

def test_data_collector():
    """Test the real-time data collector"""
    print("🧪 Testing Real-time Data Collector...")
    
    collector = RealtimeDataCollector()
    
    # Test data collection
    success = collector.collect_realtime_data()
    
    if success:
        print("\n✅ Data collection test successful!")
        
        # Get status
        status = collector.get_collection_status()
        print(f"📊 Status: {status['status']}")
        print(f"📅 Last collection: {status['last_collection']}")
        
        # Get data summary
        summary = collector.get_data_summary()
        if summary:
            print(f"📈 Data Summary:")
            print(f"   Records: {summary['total_records']}")
            print(f"   Features: {len(summary['features'])}")
            print(f"   Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        # Test forecasting preparation
        print("\n🔧 Testing Forecasting Preparation...")
        forecast_data = collector.prepare_for_forecasting()
        
        if forecast_data is not None:
            print(f"✅ Forecasting preparation successful!")
            print(f"📊 Processed data shape: {forecast_data.shape}")
        else:
            print("❌ Forecasting preparation failed")
            
    else:
        print("\n❌ Data collection test failed!")
        status = collector.get_collection_status()
        if status['errors']:
            print("Errors encountered:")
            for error in status['errors']:
                print(f"   - {error}")

if __name__ == "__main__":
    test_data_collector()
