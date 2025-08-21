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
import argparse
from typing import Optional

# Fix Meteostat compatibility issues
try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Meteostat import failed: {e}")
    METEOSTAT_AVAILABLE = False

import time
import warnings
import json
import shutil
# from logging_config import setup_logging
# from data_validation import DataValidator
warnings.filterwarnings('ignore')

# Configuration
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189
OPENWEATHER_API_KEY = "86e22ef485ce8beb1a30ba654f6c2d5a"
COLLECTION_DAYS = 1  # Collect last 24 hours for hourly updates

class DataCollector:
    def __init__(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, mode: str = "hourly"):
        """Initialize data collector

        mode: "hourly" saves to data_repositories/hourly_data, "historical" saves to data_repositories/historical_data
        """
        print("🔄 Initializing AQI Data Collection Pipeline")
        print("=" * 50)

        # Initialize dates
        self.end_date = end_date
        self.start_date = start_date

        # Create directories in selected data repository
        self.collection_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_name = "historical_data" if mode == "historical" else "hourly_data"
        self.mode = mode
        self.data_dir = os.path.join(current_dir, "data_repositories", repo_name)

        # Create required directories
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "metadata"), exist_ok=True)

        # If dates not provided, infer an incremental 3-day window from the latest stored data
        if self.start_date is None or self.end_date is None:
            self._infer_incremental_window(days=3)
        else:
            # Ensure sensible ordering and cap to now for history endpoints
            if self.end_date < self.start_date:
                self.start_date, self.end_date = self.end_date, self.start_date
            now = datetime.now()
            if self.end_date > now:
                self.end_date = now

        print(f"📍 Location: Peshawar ({PESHAWAR_LAT}, {PESHAWAR_LON})")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"⏰ Duration: {(self.end_date - self.start_date).days} days")
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
            
            # Merge into master historical raw weather data
            self._update_master_raw(df, kind="weather")

            # Save metadata (historical repo)
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hist_meta_dir = os.path.join(current_dir, "data_repositories", "historical_data", "metadata")
            os.makedirs(hist_meta_dir, exist_ok=True)
            metadata_file = os.path.join(hist_meta_dir, "weather_metadata.json")
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
            
            # Merge into master historical raw pollution data
            self._update_master_raw(df, kind="pollution")

            # Save metadata (historical repo)
            metadata = {
                "timestamp": self.collection_timestamp,
                "records": len(df),
                "start_date": df['timestamp'].min(),
                "end_date": df['timestamp'].max(),
                "missing_values": df.isnull().sum().to_dict()
            }
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hist_meta_dir = os.path.join(current_dir, "data_repositories", "historical_data", "metadata")
            os.makedirs(hist_meta_dir, exist_ok=True)
            metadata_file = os.path.join(hist_meta_dir, "pollution_metadata.json")
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
            
            # Determine processed output path for this run
            processed_file = os.path.join(self.data_dir, "processed", "merged_data.csv")

            # Update master historical merged dataset FIRST to avoid any overwrite risks
            try:
                self._update_master_merged(df, processed_file)
            except Exception as merge_exc:
                print(f"⚠️ Skipped updating master merged dataset: {merge_exc}")

            # Do not write any additional processed CSVs; master is updated above

            print(f"✅ Data processing completed")
            print(f"📊 Final dataset shape: {df.shape}")
            print(f"⏰ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None

    def _update_master_merged(self, new_df: pd.DataFrame, processed_file_path: str) -> None:
        """Append and deduplicate into master historical merged dataset.

        - Always maintains data_repositories/historical_data/processed/merged_data.csv
        - When running in 'historical' mode, the processed file already points to the master path, so skip.
        - When running in 'hourly' mode, merge hourly batch into the master historical file.
        """
        # Centralize master paths
        master_dir, master_file = self._get_master_paths()
        os.makedirs(master_dir, exist_ok=True)

        # Load existing master before writing anything
        existing_df: pd.DataFrame

        # Ensure timestamp is datetime
        new_df = new_df.copy()
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

        if os.path.exists(master_file):
            existing_df = pd.read_csv(master_file)
            # Make sure timestamp is parsed properly
            if not existing_df.empty and 'timestamp' in existing_df.columns:
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            else:
                # Create minimal structure if unexpected
                existing_df = pd.DataFrame(columns=new_df.columns)
        else:
            existing_df = pd.DataFrame(columns=new_df.columns)

        # Union columns to handle schema drift between runs
        all_columns = sorted(set(existing_df.columns) | set(new_df.columns))
        for col in all_columns:
            if col not in existing_df.columns:
                existing_df[col] = pd.NA
            if col not in new_df.columns:
                new_df[col] = pd.NA
        existing_df = existing_df[all_columns]
        new_df = new_df[all_columns]

        # Normalize timestamps to hour precision for consistent deduplication
        if 'timestamp' in existing_df.columns:
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.floor('H')
        if 'timestamp' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp']).dt.floor('H')

        existing_count = len(existing_df)
        new_count = len(new_df)

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df = (
                combined_df
                .sort_values('timestamp')
                .drop_duplicates(subset=['timestamp'], keep='last')
            )

        final_count = len(combined_df)

        # Create a backup of the master file before overwrite
        if os.path.exists(master_file):
            backup_file = os.path.join(
                master_dir,
                f"merged_data_backup_{self.collection_timestamp}.csv",
            )
            try:
                shutil.copyfile(master_file, backup_file)
                print(f"🧷 Backup created: {backup_file}")
            except Exception as backup_exc:
                print(f"⚠️ Backup failed: {backup_exc}")

        # Write with retry and atomic replace to avoid Windows file locking issues
        self._safe_write_csv(combined_df, master_file)

        # Write/refresh metadata for master historical dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_dir = os.path.join(current_dir, "data_repositories", "historical_data", "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        master_metadata_file = os.path.join(metadata_dir, "processed_metadata.json")
        master_metadata = {
            "timestamp": self.collection_timestamp,
            "records": int(len(combined_df)),
            "start_date": str(combined_df['timestamp'].min()) if not combined_df.empty else None,
            "end_date": str(combined_df['timestamp'].max()) if not combined_df.empty else None,
            "features": list(combined_df.columns),
            "missing_values": combined_df.isnull().sum().astype(int).to_dict(),
        }
        self._safe_write_json(master_metadata, master_metadata_file)

        print(
            f"📦 Master updated | existing: {existing_count:,}, new: {new_count:,}, final: {final_count:,}"
        )

    def _is_master_path(self, processed_file_path: str) -> bool:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        master_dir = os.path.join(current_dir, "data_repositories", "historical_data", "processed")
        master_file = os.path.join(master_dir, "merged_data.csv")
        return os.path.abspath(master_file) == os.path.abspath(processed_file_path)

    def _get_master_paths(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        master_dir = os.path.join(current_dir, "data_repositories", "historical_data", "processed")
        master_file = os.path.join(master_dir, "merged_data.csv")
        return master_dir, master_file

    def _read_master_df(self) -> Optional[pd.DataFrame]:
        _, master_file = self._get_master_paths()
        if os.path.exists(master_file):
            try:
                df = pd.read_csv(master_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.floor('H')
                return df
            except Exception as e:
                print(f"⚠️ Failed to read master merged file: {e}")
                return None
        return pd.DataFrame()

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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hist_raw_dir = os.path.join(current_dir, "data_repositories", "historical_data", "raw")
        hist_proc_dir = os.path.join(current_dir, "data_repositories", "historical_data", "processed")
        print("📁 Files updated:")
        print(f"   - {os.path.join(hist_raw_dir, 'weather_data.csv')}")
        print(f"   - {os.path.join(hist_raw_dir, 'pollution_data.csv')}")
        print(f"   - {os.path.join(hist_proc_dir, 'merged_data.csv')}")
        
        return True

    def _get_master_raw_paths(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        raw_dir = os.path.join(current_dir, "data_repositories", "historical_data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        weather_file = os.path.join(raw_dir, "weather_data.csv")
        pollution_file = os.path.join(raw_dir, "pollution_data.csv")
        return weather_file, pollution_file

    def _infer_incremental_window(self, days: int = 3) -> None:
        """Infer start/end dates as the next 'days' window after the latest stored timestamp.

        Priority order:
        1) Use master processed merged file's last timestamp
        2) Fall back to raw pollution, then raw weather
        If no data found, default to the last 'days' from now.
        Caps end_date to 'now'.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Master processed
        master_dir = os.path.join(current_dir, "data_repositories", "historical_data", "processed")
        master_file = os.path.join(master_dir, "merged_data.csv")
        latest_ts = None

        def read_latest_timestamp(path: str):
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path, usecols=["timestamp"])
                    if not df.empty:
                        ts = pd.to_datetime(df["timestamp"]).max()
                        if pd.notna(ts):
                            return ts.to_pydatetime()
            except Exception:
                return None
            return None

        latest_ts = read_latest_timestamp(master_file)
        if latest_ts is None:
            weather_file, pollution_file = self._get_master_raw_paths()
            latest_ts = read_latest_timestamp(pollution_file)
            if latest_ts is None:
                latest_ts = read_latest_timestamp(weather_file)

        now = datetime.now()
        if latest_ts is None:
            self.end_date = now
            self.start_date = now - timedelta(days=days)
            print(f"📅 No existing data found. Collecting default window: {self.start_date} → {self.end_date}")
            return

        # Start at the hour after the latest
        start = (pd.to_datetime(latest_ts) + pd.Timedelta(hours=1)).to_pydatetime()
        end = start + timedelta(days=days)
        if end > now:
            end = now
        self.start_date = start
        self.end_date = end
        print(f"📅 Incremental collection window: {self.start_date} → {self.end_date}")

    def _update_master_raw(self, new_df: pd.DataFrame, kind: str) -> None:
        """Append and deduplicate into master raw dataset (weather or pollution)."""
        weather_file, pollution_file = self._get_master_raw_paths()
        target_file = weather_file if kind == "weather" else pollution_file

        if new_df is None or new_df.empty:
            return

        df_new = new_df.copy()
        if 'timestamp' in df_new.columns:
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp']).dt.floor('H')

        if os.path.exists(target_file):
            try:
                existing_df = pd.read_csv(target_file)
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp']).dt.floor('H')
            except Exception:
                existing_df = pd.DataFrame(columns=df_new.columns)
        else:
            existing_df = pd.DataFrame(columns=df_new.columns)

        # Union columns
        all_columns = sorted(set(existing_df.columns) | set(df_new.columns))
        for col in all_columns:
            if col not in existing_df.columns:
                existing_df[col] = pd.NA
            if col not in df_new.columns:
                df_new[col] = pd.NA
        existing_df = existing_df[all_columns]
        df_new = df_new[all_columns]

        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
        if 'timestamp' in combined_df.columns:
            combined_df = (
                combined_df
                .sort_values('timestamp')
                .drop_duplicates(subset=['timestamp'], keep='last')
            )

        # Backup then write
        try:
            if os.path.exists(target_file):
                backup_file = target_file.replace('.csv', f"_backup_{self.collection_timestamp}.csv")
                shutil.copyfile(target_file, backup_file)
        except Exception:
            pass

        self._safe_write_csv(combined_df, target_file)

    def _safe_write_csv(self, df: pd.DataFrame, target_file: str, retries: int = 3, delay_seconds: float = 0.5) -> None:
        """Write CSV atomically with retries to mitigate 'Permission denied' on Windows.

        1) Write to a temp file in the same directory
        2) os.replace to atomically move into place
        3) On repeated failure, write to a fallback file with timestamp suffix
        """
        directory = os.path.dirname(target_file)
        os.makedirs(directory, exist_ok=True)
        temp_file = os.path.join(directory, f".{os.path.basename(target_file)}.tmp_{self.collection_timestamp}")
        fallback_file = os.path.join(directory, f"{os.path.basename(target_file)}.pending_{self.collection_timestamp}")

        # Ensure timestamp column, if present, is serializable
        df_to_write = df.copy()
        if 'timestamp' in df_to_write.columns:
            df_to_write['timestamp'] = pd.to_datetime(df_to_write['timestamp'])

        last_err = None
        for _ in range(retries):
            try:
                df_to_write.to_csv(temp_file, index=False)
                os.replace(temp_file, target_file)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(delay_seconds)
            except Exception as e:
                last_err = e
                break
        # Fallback: keep a pending file so data isn't lost
        try:
            df_to_write.to_csv(fallback_file, index=False)
            print(f"⚠️ Could not update {target_file} due to: {last_err}. Saved to {fallback_file} instead.")
        except Exception as e:
            print(f"❌ Failed to write fallback CSV for {target_file}: {e}")

    def _safe_write_json(self, data: dict, target_file: str, retries: int = 3, delay_seconds: float = 0.5) -> None:
        directory = os.path.dirname(target_file)
        os.makedirs(directory, exist_ok=True)
        temp_file = os.path.join(directory, f".{os.path.basename(target_file)}.tmp_{self.collection_timestamp}")
        fallback_file = os.path.join(directory, f"{os.path.basename(target_file)}.pending_{self.collection_timestamp}")

        payload = json.dumps(data, indent=4)
        last_err = None
        for _ in range(retries):
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(payload)
                os.replace(temp_file, target_file)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(delay_seconds)
            except Exception as e:
                last_err = e
                break
        try:
            with open(fallback_file, 'w', encoding='utf-8') as f:
                f.write(payload)
            print(f"⚠️ Could not update {target_file} due to: {last_err}. Saved to {fallback_file} instead.")
        except Exception as e:
            print(f"❌ Failed to write fallback JSON for {target_file}: {e}")

def main():
    """Run data collection pipeline"""
    parser = argparse.ArgumentParser(description="AQI data collection")
    parser.add_argument("--mode", choices=["hourly", "historical"], default="hourly", help="Save to hourly or historical repository")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD for historical mode")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD for historical mode")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start) if args.start else None
    end_dt = datetime.fromisoformat(args.end) if args.end else None

    collector = DataCollector(start_date=start_dt, end_date=end_dt, mode=args.mode)
    success = collector.run_pipeline()

    if success:
        print("\n🎉 Ready for feature engineering!")
    else:
        print("\n❌ Pipeline failed! Check error messages above.")

if __name__ == "__main__":
    main()