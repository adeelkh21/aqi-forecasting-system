"""
AQI Prediction System - Data Merger
=================================

This script:
1. Merges all collected data (historical + hourly updates)
2. Removes duplicates
3. Maintains a single consolidated dataset
4. Updates the main data files with new data
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import json
import shutil

def merge_all_data():
    """Merge all data and maintain a single consolidated dataset"""
    print("🔄 Starting Data Merger")
    print("=" * 50)
    
    # Create merged data repository directories if they don't exist
    main_dirs = [
        os.path.join('data_repositories', 'merged_data', 'raw'),
        os.path.join('data_repositories', 'merged_data', 'processed'),
        os.path.join('data_repositories', 'merged_data', 'metadata')
    ]
    for dir_path in main_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        # Find all weather and pollution data files from both repositories
        weather_files = [
            # Historical data repository
            os.path.join('data_repositories', 'historical_data', 'raw', 'historical_weather.csv'),
            # Hourly data repository
            os.path.join('data_repositories', 'hourly_data', 'raw', 'weather_data.csv')
        ]
        pollution_files = [
            # Historical data repository
            os.path.join('data_repositories', 'historical_data', 'raw', 'historical_pollution.csv'),
            # Hourly data repository
            os.path.join('data_repositories', 'hourly_data', 'raw', 'pollution_data.csv')
        ]
        
        # Filter out non-existent files
        weather_files = [f for f in weather_files if os.path.exists(f)]
        pollution_files = [f for f in pollution_files if os.path.exists(f)]
        
        print(f"\n📂 Found {len(weather_files)} weather files and {len(pollution_files)} pollution files")
        
        # Merge weather data
        weather_dfs = []
        for file in weather_files:
            df = pd.read_csv(file)
            weather_dfs.append(df)
        
        if weather_dfs:
            weather_data = pd.concat(weather_dfs, ignore_index=True)
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            weather_data = weather_data.drop_duplicates(subset=['timestamp'])
            weather_data = weather_data.sort_values('timestamp')
            
            # Save consolidated weather data
            weather_data.to_csv(os.path.join('data_repositories', 'merged_data', 'raw', 'weather_data.csv'), index=False)
            print(f"✅ Weather data merged: {len(weather_data):,} records")
        
        # Merge pollution data
        pollution_dfs = []
        for file in pollution_files:
            df = pd.read_csv(file)
            pollution_dfs.append(df)
        
        if pollution_dfs:
            pollution_data = pd.concat(pollution_dfs, ignore_index=True)
            pollution_data['timestamp'] = pd.to_datetime(pollution_data['timestamp'])
            pollution_data = pollution_data.drop_duplicates(subset=['timestamp'])
            pollution_data = pollution_data.sort_values('timestamp')
            
            # Save consolidated pollution data
            pollution_data.to_csv(os.path.join('data_repositories', 'merged_data', 'raw', 'pollution_data.csv'), index=False)
            print(f"✅ Pollution data merged: {len(pollution_data):,} records")
        
        # Merge and process final dataset
        if weather_dfs and pollution_dfs:
            # Ensure timestamps are datetime
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            pollution_data['timestamp'] = pd.to_datetime(pollution_data['timestamp'])
            
            # Round timestamps to nearest hour
            weather_data['timestamp'] = weather_data['timestamp'].dt.floor('H')
            pollution_data['timestamp'] = pollution_data['timestamp'].dt.floor('H')
            
            # Merge on timestamp
            merged_data = pd.merge(
                pollution_data,
                weather_data,
                on='timestamp',
                how='inner'
            )
            
            # Calculate AQI values
            def calculate_aqi(concentration, breakpoints):
                """Calculate AQI value given concentration and breakpoint table"""
                for _, (low_conc, high_conc, low_aqi, high_aqi) in breakpoints.items():
                    if low_conc <= concentration <= high_conc:
                        return np.interp(concentration, [low_conc, high_conc], [low_aqi, high_aqi])
                return 500  # Maximum AQI value

            # PM2.5 breakpoints (μg/m³) and corresponding AQI values
            pm25_breakpoints = {
                'Good': (0.0, 12.0, 0, 50),
                'Moderate': (12.1, 35.4, 51, 100),
                'Unhealthy for Sensitive Groups': (35.5, 55.4, 101, 150),
                'Unhealthy': (55.5, 150.4, 151, 200),
                'Very Unhealthy': (150.5, 250.4, 201, 300),
                'Hazardous': (250.5, 500.4, 301, 500)
            }
            
            # PM10 breakpoints (μg/m³) and corresponding AQI values
            pm10_breakpoints = {
                'Good': (0, 54, 0, 50),
                'Moderate': (55, 154, 51, 100),
                'Unhealthy for Sensitive Groups': (155, 254, 101, 150),
                'Unhealthy': (255, 354, 151, 200),
                'Very Unhealthy': (355, 424, 201, 300),
                'Hazardous': (425, 604, 301, 500)
            }

            # Calculate AQI based on both PM2.5 and PM10
            merged_data['aqi_pm25'] = merged_data['pm2_5'].apply(lambda x: calculate_aqi(x, pm25_breakpoints))
            merged_data['aqi_pm10'] = merged_data['pm10'].apply(lambda x: calculate_aqi(x, pm10_breakpoints))
            
            # Final AQI is the maximum of PM2.5 and PM10 AQI values
            merged_data['aqi_numeric'] = merged_data[['aqi_pm25', 'aqi_pm10']].max(axis=1)

            # Add time-based features
            merged_data['hour'] = merged_data['timestamp'].dt.hour
            merged_data['day'] = merged_data['timestamp'].dt.day
            merged_data['month'] = merged_data['timestamp'].dt.month
            merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
            merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)
            
            # Save processed data
            merged_data.to_csv(os.path.join('data_repositories', 'merged_data', 'processed', 'merged_data.csv'), index=False)
            
            # Save metadata
            metadata = {
                "last_update": datetime.now().isoformat(),
                "total_records": len(merged_data),
                "date_range": {
                    "start": merged_data['timestamp'].min(),
                    "end": merged_data['timestamp'].max()
                },
                "features": list(merged_data.columns),
                "data_sources": {
                    "weather_files": len(weather_files),
                    "pollution_files": len(pollution_files)
                },
                "missing_values": merged_data.isnull().sum().to_dict()
            }
            
            with open(os.path.join('data_repositories', 'merged_data', 'metadata', 'dataset_info.json'), 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print("\n📊 Final Dataset Summary:")
            print(f"Total records: {len(merged_data):,}")
            print(f"Date range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
            print(f"Features: {len(merged_data.columns)}")
            
            # Data from both repositories merged successfully
            
            return True
            
    except Exception as e:
        print(f"❌ Error merging data: {str(e)}")
        return False

def main():
    """Run data merger"""
    success = merge_all_data()
    
    if success:
        print("\n✅ Data merger completed successfully!")
        print("📁 Final data structure:")
        print("   data_repositories/")
        print("   ├── hourly_data/")
        print("   │   ├── raw/")
        print("   │   │   ├── weather_data.csv")
        print("   │   │   └── pollution_data.csv")
        print("   │   ├── processed/")
        print("   │   │   └── merged_data.csv")
        print("   │   └── metadata/")
        print("   │       └── dataset_info.json")
        print("   └── merged_data/")
        print("       ├── raw/")
        print("       │   ├── weather_data.csv")
        print("       │   └── pollution_data.csv")
        print("       ├── processed/")
        print("       │   └── merged_data.csv")
        print("       └── metadata/")
        print("           └── dataset_info.json")
    else:
        print("\n❌ Data merger failed! Check error messages above.")

if __name__ == "__main__":
    main()
