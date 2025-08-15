"""
AQI Prediction System - Historical Data Collection
===============================================

This script collects 150 days of historical weather and pollution data:
- Weather data from Meteostat API
- Pollution data from OpenWeatherMap API
- Saves to a separate historical data repository
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
from logging_config import setup_logging
from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = "86e22ef485ce8beb1a30ba654f6c2d5a"
COLLECTION_DAYS = 150  # Collect 150 days of historical data

class HistoricalDataCollector:
    def __init__(self):
        """Initialize historical data collector"""
        print("🔄 Initializing Historical Data Collection")
        print("=" * 50)
        
        # Initialize dates
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=COLLECTION_DAYS)
        
        # Create directories in historical data repository
        self.collection_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join("data_repositories", "historical_data")
        
        # Create required directories
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "metadata"), exist_ok=True)
        
        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {COLLECTION_DAYS} days")

    def fetch_weather_data(self):
        """Fetch historical weather data from Meteostat"""
        print("\n🌤️ Fetching Historical Weather Data")
        print("-" * 40)
        
        try:
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            data = Hourly(location, self.start_date, self.end_date)
            df = data.fetch()
            
            if df is None or df.empty:
                print("❌ No weather data received!")
                return None
            
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
            
            # Save raw data
            weather_file = os.path.join(self.data_dir, "raw", "historical_weather.csv")
            df.to_csv(weather_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "historical_weather_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Historical weather data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return None

    def fetch_pollution_data(self):
        """Fetch historical pollution data from OpenWeatherMap API"""
        print("\n🏭 Fetching Historical Pollution Data")
        print("-" * 40)
        
        try:
            # Split into smaller time chunks to avoid API limits
            chunk_size = timedelta(days=5)
            current_start = self.start_date
            all_results = []
            
            while current_start < self.end_date:
                current_end = min(current_start + chunk_size, self.end_date)
                
                print(f"\n📅 Fetching chunk: {current_start.date()} to {current_end.date()}")
                
                start_timestamp = int(current_start.timestamp())
                end_timestamp = int(current_end.timestamp())
                
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
                    # Wait before retrying
                    time.sleep(2)
                    continue
                
                data = response.json()
                
                for item in data.get('list', []):
                    record = {
                        "timestamp": datetime.utcfromtimestamp(item['dt']),
                        "aqi_category": item['main']['aqi'],
                        **item['components']
                    }
                    all_results.append(record)
                
                # Move to next chunk
                current_start = current_end
                # Wait between chunks to respect API rate limits
                time.sleep(1)
            
            if not all_results:
                print("❌ No pollution data collected!")
                return None
            
            df = pd.DataFrame(all_results)
            df = df.sort_values('timestamp')
            
            # Save raw data
            pollution_file = os.path.join(self.data_dir, "raw", "historical_pollution.csv")
            df.to_csv(pollution_file, index=False)
            
            # Save metadata
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            metadata_file = os.path.join(self.data_dir, "metadata", "historical_pollution_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"\n✅ Historical pollution data collected: {len(df):,} records")
            print(f"📊 Features: {', '.join(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching pollution data: {str(e)}")
            return None

    def merge_and_process_data(self, weather_df, pollution_df):
        """Merge historical weather and pollution data"""
        print("\n🔄 Processing and Merging Historical Data")
        print("-" * 40)
        
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
            
            # Save processed data
            processed_file = os.path.join(self.data_dir, "processed", "historical_merged.csv")
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
            
            metadata_file = os.path.join(self.data_dir, "metadata", "historical_merged_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"✅ Historical data processing completed")
            print(f"📊 Final dataset shape: {df.shape}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None

    def run_pipeline(self):
        """Run complete historical data collection pipeline"""
        print("\n🚀 Starting Historical Data Collection Pipeline")
        print("=" * 50)
        
        # Step 1: Fetch historical weather data
        weather_df = self.fetch_weather_data()
        if weather_df is None:
            return False
        
        # Step 2: Fetch historical pollution data
        pollution_df = self.fetch_pollution_data()
        if pollution_df is None:
            return False
        
        # Step 3: Process and merge data
        final_df = self.merge_and_process_data(weather_df, pollution_df)
        if final_df is None:
            return False
        
        print("\n✅ Historical Data Collection Completed Successfully!")
        print("=" * 50)
        print("📁 Files saved:")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'historical_weather.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'raw', 'historical_pollution.csv')}")
        print(f"   - {os.path.join(self.data_dir, 'processed', 'historical_merged.csv')}")
        
        return True

def main():
    """Run historical data collection pipeline"""
    collector = HistoricalDataCollector()
    success = collector.run_pipeline()
    
    if success:
        print("\n🎉 Ready for feature engineering!")
    else:
        print("\n❌ Pipeline failed! Check error messages above.")

if __name__ == "__main__":
    main()