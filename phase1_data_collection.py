"""
AQI Prediction System - Hourly Data Collection Pipeline
=============================================

This script collects hourly weather and pollution data for AQI prediction:
- Weather data from Meteostat API
- Pollution data from OpenWeatherMap API
- Converts categorical AQI to numerical values

Author: Data Science Team
Date: 2024-03-09
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import time
import warnings
import json
# from logging_config import setup_logging
# from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = "86e22ef485ce8beb1a30ba654f6c2d5a"
COLLECTION_DAYS = 1  # Collect last 24 hours for hourly updates

class DataCollector:
    def __init__(self):
        """Initialize data collector"""
        print("🔄 Initializing AQI Data Collection Pipeline")
        print("=" * 50)
        
        # Initialize dates (use longer period to ensure data availability)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=max(COLLECTION_DAYS, 7))
        
        # Create directories in hourly data repository
        self.collection_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join("data_repositories", "hourly_data")
        
        # Create required directories
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "metadata"), exist_ok=True)
        
        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {COLLECTION_DAYS} days")
        print(f"📂 Data Directory: {self.data_dir}")

    def fetch_weather_data(self):
        """Fetch weather data from Meteostat with fallback to Daily data"""
        print("\n🌤️ Fetching Weather Data")
        print("-" * 30)
        
        try:
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            
            # Try Hourly data first
            try:
                print("   🔄 Attempting to fetch hourly data...")
                data = Hourly(location, self.start_date, self.end_date)
                df = data.fetch()
                
                if df is not None and not df.empty:
                    df.reset_index(inplace=True)
                    df.rename(columns={
                        'time': 'timestamp',
                        'temp': 'temperature',
                        'dwpt': 'dew_point',
                        'rhum': 'relative_humidity',
                        'prcp': 'precipitation',
                        'wdir': 'wind_direction',
                        'wspd': 'wind_speed',
                        'pres': 'pressure'
                    }, inplace=True)
                    print("   ✅ Hourly data fetched successfully")
                else:
                    raise Exception("No hourly data received")
                    
            except Exception as hourly_error:
                print(f"   ⚠️ Hourly data failed: {str(hourly_error)[:100]}...")
                print("   🔄 Falling back to daily data...")
                
                # Fallback to Daily data
                from meteostat import Daily
                data = Daily(location, self.start_date, self.end_date)
                df = data.fetch()
                
                if df is None or df.empty:
                    print("❌ No daily data received either!")
                    return None
                
                df.reset_index(inplace=True)
                # Daily data has different column names
                df.rename(columns={
                    'time': 'timestamp',
                    'tavg': 'temperature',  # Use average temperature
                    'prcp': 'precipitation',
                    'wdir': 'wind_direction',
                    'wspd': 'wind_speed',
                    'pres': 'pressure'
                }, inplace=True)
                
                # Handle additional daily data columns
                if 'tmin' in df.columns and 'tmax' in df.columns:
                    df['temperature_range'] = df['tmax'] - df['tmin']
                else:
                    df['temperature_range'] = 0
                
                # Add missing columns with reasonable defaults for daily data
                df['dew_point'] = df['temperature'] - 5  # Approximate dew point
                df['relative_humidity'] = 60  # Default humidity
                
                print("   ✅ Daily data fetched successfully (with estimated values)")
                
                # Upsample daily data to hourly to match pollution data frequency
                print("   🔄 Upsampling daily data to hourly...")
                df.set_index('timestamp', inplace=True)
                
                # Resample to hourly and forward fill missing values
                df_hourly = df.resample('H').ffill()
                
                # Reset index to get timestamp as column again
                df_hourly.reset_index(inplace=True)
                
                print(f"   ✅ Upsampled to {len(df_hourly)} hourly records")
                df = df_hourly
            
            if df is None or df.empty:
                print("❌ No weather data received!")
                return None
            
            # Save raw data
            weather_file = os.path.join(self.data_dir, "raw", "weather_data.csv")
            df.to_csv(weather_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "weather_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Weather data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return None

    def fetch_pollution_data(self):
        """Fetch pollution data from OpenWeatherMap API"""
        print("\n🏭 Fetching Pollution Data")
        print("-" * 30)
        
        try:
            end_timestamp = int(self.end_date.timestamp())
            start_timestamp = int(self.start_date.timestamp())
            
            url = (
                f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
                f"lat={PESHAWAR_LAT}&lon={PESHAWAR_LON}&"
                f"start={start_timestamp}&end={end_timestamp}&"
                f"appid={OPENWEATHER_API_KEY}"
            )
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
            data = response.json()
            results = []
            
            for item in data.get('list', []):
                record = {
                    "timestamp": datetime.utcfromtimestamp(item['dt']),
                    "aqi_category": item['main']['aqi'],
                    **item['components']
                }
                results.append(record)
            
            if not results:
                print("❌ No pollution data collected!")
                return None
            
            df = pd.DataFrame(results)
            df = df.sort_values('timestamp')
            
            # Save raw data
            pollution_file = os.path.join(self.data_dir, "raw", "pollution_data.csv")
            df.to_csv(pollution_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "pollution_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Pollution data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching pollution data: {str(e)}")
            return None

    def merge_and_process_data(self, weather_df, pollution_df):
        """Merge weather and pollution data, process for modeling"""
        print("\n🔄 Processing and Merging Data")
        print("-" * 30)
        
        try:
            # Ensure timestamps are datetime
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
            pollution_df['timestamp'] = pd.to_datetime(pollution_df['timestamp'])
            
            # Round timestamps to nearest hour
            weather_df['timestamp'] = weather_df['timestamp'].dt.floor('H')
            pollution_df['timestamp'] = pollution_df['timestamp'].dt.floor('H')
            
            # Merge on timestamp
            df = pd.merge(
                pollution_df,
                weather_df,
                on='timestamp',
                how='inner'
            )
            
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add advanced cyclical features (matching training data)
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            
            # Add missing features with reasonable defaults
            if 'coco' not in df.columns:
                df['coco'] = 0  # Default cloud coverage
            
            # Save processed data
            processed_file = os.path.join(self.data_dir, "processed", "merged_data.csv")
            df.to_csv(processed_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "features": list(df.columns),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "processed_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Data processing completed")
            print(f"📊 Final dataset shape: {df.shape}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None

    def run_pipeline(self):
        """Run complete data collection pipeline"""
        print("\n🚀 Starting Data Collection Pipeline")
        print("=" * 50)
        
        # Step 1: Fetch weather data
        weather_df = self.fetch_weather_data()
        if weather_df is None:
            return False
        
        # Step 2: Fetch pollution data
        pollution_df = self.fetch_pollution_data()
        if pollution_df is None:
            return False
        
        # Step 3: Process and merge data
        final_df = self.merge_and_process_data(weather_df, pollution_df)
        if final_df is None:
            return False
        
        print("\n✅ Data Collection Pipeline Completed Successfully!")
        print("=" * 50)
        print("📁 Files saved:")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'weather_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'pollution_data.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'processed', 'merged_data.csv')}")
        
        return True

def main():
    """Run data collection pipeline"""
    collector = DataCollector()
    success = collector.run_pipeline()
    
    if success:
        print("\n🎉 Ready for feature engineering!")
    else:
        print("\n❌ Pipeline failed! Check error messages above.")

if __name__ == "__main__":
    main()