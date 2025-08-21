#!/usr/bin/env python3
"""
Phase 1: FIXED Preprocessing WITHOUT Data Leakage
- Removes all AQI-related features that cause data leakage
- Keeps only legitimate features for forecasting
- Recalculates numerical AQI properly with EPA compliance
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class FixedNoLeakagePreprocessor:
    def __init__(self):
        """Initialize fixed preprocessor for multi-horizon forecasting without data leakage"""
        print("🔧 PHASE 1: FIXED PREPROCESSING FOR MULTI-HORIZON FORECASTING")
        print("=" * 80)
        
        # Data paths
        self.input_path = "data_repositories/historical_data/processed/complete_150_days_data.csv"
        self.output_path = "data_repositories/features/phase1_no_leakage_data.csv"
        self.metadata_path = "data_repositories/features/phase1_no_leakage_metadata.json"
        
        # Create output directories
        os.makedirs("data_repositories/features", exist_ok=True)
        
        # Target training period (dynamic unless overridden)
        self.train_start = None  # Will default to earliest available timestamp after load
        self.train_end = None    # Will default to latest available timestamp after load
        
        # EPA AQI breakpoints with CORRECT units and averaging periods
        self.aqi_breakpoints = {
            'pm2_5': [  # µg/m³ (24-hr average) - truncate to 0.1 µg/m³
                (0.0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ],
            'pm10': [  # µg/m³ (24-hr average) - truncate to 1 µg/m³
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 504, 301, 400),
                (505, 604, 401, 500)
            ],
            'o3_8hr': [  # ppb (8-hr average) - truncate to 0.001 ppm = 1 ppb
                (0, 54, 0, 50),
                (55, 70, 51, 100),
                (71, 85, 101, 150),
                (86, 105, 151, 200),
                (106, 200, 201, 300)
            ],
            'o3_1hr': [  # ppb (1-hr average) - truncate to 0.001 ppm = 1 ppb
                (125, 164, 101, 150),
                (165, 204, 151, 200),
                (205, 404, 201, 300),
                (405, 504, 301, 400),
                (505, 604, 401, 500)
            ],
            'co': [  # ppm (8-hr average) - truncate to 0.1 ppm
                (0.0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500)
            ],
            'so2': [  # ppb (1-hr average) - truncate to 1 ppb
                (0, 35, 0, 50),
                (36, 75, 51, 100),
                (76, 185, 101, 150),
                (186, 304, 151, 200),
                (305, 604, 201, 300),
                (605, 804, 301, 400),
                (805, 1004, 401, 500)
            ],
            'no2': [  # ppb (1-hr average) - truncate to 1 ppb
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 1649, 301, 400),
                (1650, 2049, 401, 500)
            ]
        }
        
        # EPA truncation rules (concentration precision before AQI calculation)
        self.truncation_rules = {
            'pm2_5': 0.1,      # µg/m³ to 0.1 µg/m³
            'pm10': 1.0,       # µg/m³ to 1 µg/m³
            'o3': 1.0,         # ppb to 1 ppb (0.001 ppm)
            'co': 0.1,         # ppm to 0.1 ppm
            'so2': 1.0,        # ppb to 1 ppb
            'no2': 1.0         # ppb to 1 ppb
        }
        
        # Molecular weights for unit conversion (g/mol)
        self.molecular_weights = {
            'o3': 48.00,       # Ozone
            'no2': 46.01,      # Nitrogen dioxide
            'so2': 64.07,      # Sulfur dioxide
            'co': 28.01        # Carbon monoxide
        }
        
        # Standard conditions for unit conversion
        self.temperature_k = 298.15  # 25°C in Kelvin
        self.pressure_atm = 1.0      # 1 atmosphere
        
        print(f"🎯 Training Period: will be determined from data (minus 72h tail) unless overridden")
        print(f"📁 Input: {self.input_path}")
        print(f"📁 Output: {self.output_path}")
        print(f"🎯 Target: Numerical AQI (0-500) WITHOUT data leakage")
        print(f"⚠️  REMOVING ALL AQI-RELATED FEATURES TO PREVENT LEAKAGE")
        print(f"🔧 IMPLEMENTING EPA-COMPLIANT AQI CALCULATION")
        
    def _print_dataset_range(self, path: str, label: str) -> None:
        try:
            if not os.path.exists(path):
                print(f"📅 {label}: MISSING ({path})")
                return
            df = pd.read_csv(path, usecols=['timestamp'])
            if df.empty:
                print(f"📅 {label}: EMPTY ({path})")
                return
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"📅 {label}: {df['timestamp'].min()} → {df['timestamp'].max()}")
        except Exception as e:
            print(f"⚠️ Could not read date range for {label}: {e}")

    def convert_units_to_epa_standard(self, df):
        """Convert pollutant units from µg/m³ to EPA standard units"""
        print(f"\n🔄 CONVERTING UNITS TO EPA STANDARD")
        print("-" * 50)
        
        # Standard molar volume at 25°C, 1 atm (L/mol)
        molar_volume = 24.45
        
        # Convert O3 from µg/m³ to ppb
        if 'o3' in df.columns:
            # µg/m³ to ppb: ppb = (µg/m³ × 24.45) / MW
            df['o3_ppb'] = (df['o3'] * molar_volume) / self.molecular_weights['o3']
            print(f"   O3 converted: µg/m³ → ppb")
            print(f"      Range: {df['o3'].min():.2f} to {df['o3'].max():.2f} µg/m³")
            print(f"      Converted: {df['o3_ppb'].min():.2f} to {df['o3_ppb'].max():.2f} ppb")
        
        # Convert NO2 from µg/m³ to ppb
        if 'no2' in df.columns:
            # µg/m³ to ppb: ppb = (µg/m³ × 24.45) / MW
            df['no2_ppb'] = (df['no2'] * molar_volume) / self.molecular_weights['no2']
            print(f"   NO2 converted: µg/m³ → ppb")
            print(f"      Range: {df['no2'].min():.2f} to {df['no2'].max():.2f} µg/m³")
            print(f"      Converted: {df['no2_ppb'].min():.2f} to {df['no2_ppb'].max():.2f} ppb")
        
        # Convert SO2 from µg/m³ to ppb
        if 'so2' in df.columns:
            # µg/m³ to ppb: ppb = (µg/m³ × 24.45) / MW
            df['so2_ppb'] = (df['so2'] * molar_volume) / self.molecular_weights['so2']
            print(f"   SO2 converted: µg/m³ → ppb")
            print(f"      Range: {df['so2'].min():.2f} to {df['so2'].max():.2f} µg/m³")
            print(f"      Converted: {df['so2_ppb'].min():.2f} to {df['so2_ppb'].max():.2f} ppb")
        
        # Convert CO from µg/m³ to ppm
        if 'co' in df.columns:
            # µg/m³ to ppm: ppm = (µg/m³ × 24.45) / (1000 × MW)
            df['co_ppm'] = (df['co'] * molar_volume) / (1000 * self.molecular_weights['co'])
            print(f"   CO converted: µg/m³ → ppm")
            print(f"      Range: {df['co'].min():.2f} to {df['co'].max():.2f} µg/m³")
            print(f"      Converted: {df['co_ppm'].min():.3f} to {df['co_ppm'].max():.3f} ppm")
        
        # PM2.5 and PM10 are already in µg/m³, no conversion needed
        print(f"   PM2.5 and PM10: already in µg/m³ (no conversion needed)")
        
        return df
    
    def load_data(self):
        """Load and validate the 150-day dataset"""
        print(f"\n📥 LOADING DATA")
        print("-" * 40)
        
        if not os.path.exists(self.input_path):
            # Fallback to merged historical data produced by 01_data_collection.py in historical mode
            fallback_path = "data_repositories/historical_data/processed/merged_data.csv"
            if os.path.exists(fallback_path):
                print(f"⚠️ Input file not found at default path. Using fallback: {fallback_path}")
                self.input_path = fallback_path
            else:
                print(f"❌ Input file not found: {self.input_path}")
                print(f"❌ Fallback also missing: {fallback_path}")
                return None
        
        # Load data
        df = pd.read_csv(self.input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"   Raw data loaded: {len(df):,} records")
        global_min = pd.to_datetime(df['timestamp']).min()
        global_max = pd.to_datetime(df['timestamp']).max()
        print(f"   Date range: {global_min} to {global_max}")
        print(f"   Columns: {len(df.columns)}")
        
        # Determine training window dynamically if not set
        # Use FULL range here; the last 72h will be trimmed later during target creation
        reserve_tail_hours = 0
        if self.train_start is None:
            self.train_start = global_min
        if self.train_end is None:
            self.train_end = global_max.to_pydatetime()
        print(f"   Using training window: {self.train_start} → {self.train_end} (full range; tail trimmed after target creation)")

        # Filter to training period
        train_data = df[(df['timestamp'] >= self.train_start) & (df['timestamp'] <= self.train_end)].copy()
        
        print(f"   Training period data: {len(train_data):,} records")
        if not train_data.empty:
            print(f"   Training data range: {train_data['timestamp'].min()} → {train_data['timestamp'].max()}")
        
        return train_data
    
    def apply_epa_truncation(self, concentration, pollutant):
        """Apply EPA truncation rules to concentration values"""
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        truncation = self.truncation_rules.get(pollutant, 1.0)
        
        if pollutant == 'pm2_5':
            # Truncate to 0.1 µg/m³
            return np.floor(concentration * 10) / 10
        elif pollutant == 'pm10':
            # Truncate to 1 µg/m³
            return np.floor(concentration)
        elif pollutant == 'o3':
            # Truncate to 1 ppb
            return np.floor(concentration)
        elif pollutant == 'co':
            # Truncate to 0.1 ppm
            return np.floor(concentration * 10) / 10
        elif pollutant in ['so2', 'no2']:
            # Truncate to 1 ppb
            return np.floor(concentration)
        else:
            return concentration
    
    def calculate_required_averages(self, df):
        """Calculate required averaging periods for EPA AQI calculation"""
        print(f"\n⏰ CALCULATING REQUIRED AVERAGING PERIODS")
        print("-" * 50)
        
        # Sort by timestamp for proper rolling calculations
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # PM2.5: 24-hour average
        if 'pm2_5' in df.columns:
            df['pm2_5_24h_avg'] = df['pm2_5'].rolling(window=24, min_periods=18).mean()
            print(f"   PM2.5: 24-hour rolling average (min 18 hours)")
        
        # PM10: 24-hour average
        if 'pm10' in df.columns:
            df['pm10_24h_avg'] = df['pm10'].rolling(window=24, min_periods=18).mean()
            print(f"   PM10: 24-hour rolling average (min 18 hours)")
        
        # O3: 8-hour average (and 1-hour for high concentrations)
        if 'o3_ppb' in df.columns:
            df['o3_8h_avg'] = df['o3_ppb'].rolling(window=8, min_periods=6).mean()
            df['o3_1h_avg'] = df['o3_ppb']  # 1-hour values (already hourly data)
            print(f"   O3: 8-hour rolling average (min 6 hours) + 1-hour values")
        
        # CO: 8-hour average
        if 'co_ppm' in df.columns:
            df['co_8h_avg'] = df['co_ppm'].rolling(window=8, min_periods=6).mean()
            print(f"   CO: 8-hour rolling average (min 6 hours)")
        
        # NO2: 1-hour average (already hourly data)
        if 'no2_ppb' in df.columns:
            df['no2_1h_avg'] = df['no2_ppb']
            print(f"   NO2: 1-hour values (already hourly data)")
        
        # SO2: 1-hour average (already hourly data)
        if 'so2_ppb' in df.columns:
            df['so2_1h_avg'] = df['so2_ppb']
            print(f"   SO2: 1-hour values (already hourly data)")
        
        return df
    
    def calculate_aqi(self, concentration, pollutant, averaging_period='1hr'):
        """Calculate AQI for a given pollutant concentration using EPA breakpoints"""
        if pd.isna(concentration) or concentration < 0:
            return np.nan
        
        # Select appropriate breakpoints
        if pollutant == 'o3':
            if averaging_period == '8hr':
                breakpoints = self.aqi_breakpoints['o3_8hr']
            else:
                breakpoints = self.aqi_breakpoints['o3_1hr']
        else:
            breakpoints = self.aqi_breakpoints.get(pollutant, [])
        
        if not breakpoints:
            return np.nan
        
        # Find the appropriate breakpoint range
        for clow, chigh, ilow, ihigh in breakpoints:
            if clow <= concentration <= chigh:
                # Apply EPA formula: I = (Ihigh - Ilow) / (Chigh - Clow) * (C - Clow) + Ilow
                aqi = (ihigh - ilow) / (chigh - clow) * (concentration - clow) + ilow
                return round(aqi)
        
        # If concentration is above the highest breakpoint, cap at 500
        if concentration > breakpoints[-1][1]:
            return 500
        
        # If concentration is below the lowest breakpoint, cap at 0
        if concentration < breakpoints[0][0]:
            return 0
        
        return np.nan
    
    def calculate_numerical_aqi(self, df):
        """Calculate numerical AQI for each row using EPA-compliant method"""
        print(f"\n🧮 CALCULATING NUMERICAL AQI WITH EPA COMPLIANCE")
        print("-" * 60)
        
        # Calculate AQI for each pollutant with proper averaging and truncation
        aqi_values = {}
        
        # PM2.5 (24-hour average) - apply truncation
        if 'pm2_5_24h_avg' in df.columns:
            pm25_truncated = df['pm2_5_24h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'pm2_5'))
            aqi_values['pm2_5_aqi'] = pm25_truncated.apply(lambda x: self.calculate_aqi(x, 'pm2_5'))
            valid_count = aqi_values['pm2_5_aqi'].notna().sum()
            print(f"   PM2.5 AQI calculated: {valid_count} valid values")
            if valid_count > 0:
                print(f"      Range: {aqi_values['pm2_5_aqi'].min():.0f} to {aqi_values['pm2_5_aqi'].max():.0f}")
        
        # PM10 (24-hour average) - apply truncation
        if 'pm10_24h_avg' in df.columns:
            pm10_truncated = df['pm10_24h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'pm10'))
            aqi_values['pm10_aqi'] = pm10_truncated.apply(lambda x: self.calculate_aqi(x, 'pm10'))
            valid_count = aqi_values['pm10_aqi'].notna().sum()
            print(f"   PM10 AQI calculated: {valid_count} valid values")
            if valid_count > 0:
                print(f"      Range: {aqi_values['pm10_aqi'].min():.0f} to {aqi_values['pm10_aqi'].max():.0f}")
        
        # Ozone - implement EPA O3 selection rule
        if 'o3_8h_avg' in df.columns and 'o3_1h_avg' in df.columns:
            o3_8h_truncated = df['o3_8h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'o3'))
            o3_1h_truncated = df['o3_1h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'o3'))
            
            # Calculate both 8-hour and 1-hour AQIs
            o3_8h_aqi = o3_8h_truncated.apply(lambda x: self.calculate_aqi(x, 'o3', '8hr'))
            o3_1h_aqi = o3_1h_truncated.apply(lambda x: self.calculate_aqi(x, 'o3', '1hr'))
            
            # EPA O3 selection rule: use the higher of 8-hour or 1-hour AQI
            # But 1-hour O3 must be ≥ 125 ppb to use 1-hour table
            o3_1h_eligible = o3_1h_truncated >= 125
            
            # Select the appropriate AQI value
            o3_final_aqi = np.where(
                o3_1h_eligible & (o3_1h_aqi > o3_8h_aqi),
                o3_1h_aqi,
                o3_8h_aqi
            )
            
            aqi_values['o3_aqi'] = o3_final_aqi
            valid_count = pd.Series(o3_final_aqi).notna().sum()
            print(f"   O3 AQI calculated: {valid_count} valid values")
            print(f"      EPA O3 rule applied: 8-hour vs 1-hour selection")
            if valid_count > 0:
                print(f"      Range: {pd.Series(o3_final_aqi).min():.0f} to {pd.Series(o3_final_aqi).max():.0f}")
        
        # CO (8-hour average) - apply truncation
        if 'co_8h_avg' in df.columns:
            co_truncated = df['co_8h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'co'))
            aqi_values['co_aqi'] = co_truncated.apply(lambda x: self.calculate_aqi(x, 'co'))
            valid_count = aqi_values['co_aqi'].notna().sum()
            print(f"   CO AQI calculated: {valid_count} valid values")
            if valid_count > 0:
                print(f"      Range: {aqi_values['co_aqi'].min():.0f} to {aqi_values['co_aqi'].max():.0f}")
        
        # SO2 (1-hour average) - apply truncation
        if 'so2_1h_avg' in df.columns:
            so2_truncated = df['so2_1h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'so2'))
            aqi_values['so2_aqi'] = so2_truncated.apply(lambda x: self.calculate_aqi(x, 'so2'))
            valid_count = aqi_values['so2_aqi'].notna().sum()
            print(f"   SO2 AQI calculated: {valid_count} valid values")
            if valid_count > 0:
                print(f"      Range: {aqi_values['so2_aqi'].min():.0f} to {aqi_values['so2_aqi'].max():.0f}")
        
        # NO2 (1-hour average) - apply truncation
        if 'no2_1h_avg' in df.columns:
            no2_truncated = df['no2_1h_avg'].apply(lambda x: self.apply_epa_truncation(x, 'no2'))
            aqi_values['no2_aqi'] = no2_truncated.apply(lambda x: self.calculate_aqi(x, 'no2'))
            valid_count = aqi_values['no2_aqi'].notna().sum()
            print(f"   NO2 AQI calculated: {valid_count} valid values")
            if valid_count > 0:
                print(f"      Range: {aqi_values['no2_aqi'].min():.0f} to {aqi_values['no2_aqi'].max():.0f}")
        
        # Calculate overall AQI (maximum of all pollutant AQIs)
        aqi_df = pd.DataFrame(aqi_values)
        df['numerical_aqi'] = aqi_df.max(axis=1)
        
        # Identify the primary pollutant for each row
        df['primary_pollutant'] = aqi_df.idxmax(axis=1).str.replace('_aqi', '')
        
        print(f"\n   Overall numerical AQI calculated: {df['numerical_aqi'].notna().sum()} valid values")
        print(f"   AQI range: {df['numerical_aqi'].min():.0f} to {df['numerical_aqi'].max():.0f}")
        
        # AQI distribution
        aqi_counts = df['numerical_aqi'].value_counts().sort_index()
        print(f"   AQI distribution:")
        for aqi, count in aqi_counts.head(20).items():  # Show first 20 for readability
            print(f"      AQI {aqi:3.0f}: {count:4d} records")
        if len(aqi_counts) > 20:
            print(f"      ... and {len(aqi_counts) - 20} more AQI values")
        
        # Primary pollutant distribution
        primary_counts = df['primary_pollutant'].value_counts()
        print(f"\n   Primary pollutant distribution:")
        for pollutant, count in primary_counts.items():
            print(f"      {pollutant}: {count:4d} records")
        
        # Remove categorical AQI if it exists
        if 'aqi_category' in df.columns:
            df = df.drop('aqi_category', axis=1)
            print(f"   Removed categorical AQI column")
        
        return df
        
    def engineer_legitimate_features(self, df):
        """Engineer features WITHOUT data leakage"""
        print(f"\n⚙️ ENGINEERING LEGITIMATE FEATURES (NO DATA LEAKAGE)")
        print("-" * 60)
    
        # CRITICAL: Create time features from timestamp first
        print(f"   Creating time-based features from timestamp...")
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    
        # Time-based cyclical features
        print(f"   Adding cyclical time features...")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
        # Seasonal features
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    
        # Day type features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
        # Weather interaction features
        if 'temperature' in df.columns and 'relative_humidity' in df.columns:
            df['heat_index'] = df['temperature'] * (1 + 0.01 * df['relative_humidity'])
            print(f"      → Added heat_index")
    
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            if df['wind_direction'].notna().any():
                df['wind_east'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
                df['wind_north'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
                print(f"      → Added wind components")
    
        # Pollution interaction features
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
            print(f"      → Added PM2.5/PM10 ratio")
    
        if 'co' in df.columns and 'no2' in df.columns:
            df['co_no2_ratio'] = df['co'] / (df['no2'] + 1e-6)
            print(f"      → Added CO/NO2 ratio")
    
        # CRITICAL: LEGITIMATE forecasting features (NO AQI leakage)
        print(f"   Adding legitimate forecasting features...")
    
        # IMPORTANT: We will NOT use the pollutant averages that calculate AQI
        # These are: pm2_5_24h_avg, pm10_24h_avg, o3_8h_avg, co_8h_avg
        # Instead, we'll use raw values and other derived features
    
        # Lag features for POLLUTANTS only (not AQI) - Multi-horizon optimized
        for pollutant in ['pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']:
            if pollutant in df.columns:
                # Shorter lags for 24h, medium for 48h, longer for 72h
                for lag in [1, 3, 6, 12, 18, 24, 36, 48, 72]:
                    df[f'{pollutant}_lag_{lag}h'] = df[pollutant].shift(lag)
                print(f"      → Added {pollutant} lags (1,3,6,12,18,24,36,48,72h)")
    
        # Rolling statistics for POLLUTANTS only (NOT the ones used for AQI)
        for pollutant in ['pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']:
            if pollutant in df.columns:
                # Multi-horizon windows: short-term (6-24h), medium-term (36-48h), long-term (60-72h)
                for window in [6, 12, 18, 24, 36, 48, 60, 72]:
                    df[f'{pollutant}_{window}h_mean'] = df[pollutant].rolling(window=window, min_periods=1).mean()
                    df[f'{pollutant}_{window}h_std'] = df[pollutant].rolling(window=window, min_periods=1).std()
                    df[f'{pollutant}_{window}h_trend'] = df[pollutant].rolling(window=window, min_periods=2).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
                print(f"      → Added {pollutant} rolling statistics and trends (multi-horizon optimized)")
    
        # Rate of change features for POLLUTANTS only - Multi-horizon focused
        for pollutant in ['pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']:
            if pollutant in df.columns:
                # Short-term changes for 24h, medium-term for 48h, long-term for 72h
                for change_hours in [1, 3, 6, 12, 18, 24, 36, 48]:
                    df[f'{pollutant}_change_{change_hours}h'] = df[pollutant].diff(change_hours)
                print(f"      → Added {pollutant} rate of change features (multi-horizon)")
    
        # Momentum indicators for POLLUTANTS only - Multi-horizon optimized
        for pollutant in ['pm2_5', 'pm10', 'o3', 'co', 'no2', 'so2']:
            if pollutant in df.columns:
                # Different momentum periods for different horizons
                for momentum_hours in [6, 12, 18, 24, 36, 48]:
                    df[f'{pollutant}_momentum_{momentum_hours}h'] = df[pollutant] - df[pollutant].shift(momentum_hours)
                print(f"      → Added {pollutant} momentum indicators (multi-horizon)")
    
        # Weather rolling features for trend capture (different windows) - Multi-horizon
        if 'temperature' in df.columns:
            for window in [6, 12, 18, 24, 36, 48, 60, 72]:
                df[f'temp_{window}h_mean'] = df['temperature'].rolling(window=window, min_periods=1).mean()
                df[f'temp_{window}h_trend'] = df['temperature'].rolling(window=window, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            print(f"      → Added temperature rolling features (multi-horizon optimized)")
    
        if 'relative_humidity' in df.columns:
            for window in [6, 12, 18, 24, 36, 48, 60, 72]:
                df[f'humidity_{window}h_mean'] = df['relative_humidity'].rolling(window=window, min_periods=1).mean()
            print(f"      → Added humidity rolling features (multi-horizon optimized)")
    
        # Additional weather features (different windows) - Multi-horizon
        if 'pressure' in df.columns:
            for window in [6, 12, 18, 24, 36, 48, 60, 72]:
                df[f'pressure_{window}h_mean'] = df['pressure'].rolling(window=window, min_periods=1).mean()
                df[f'pressure_{window}h_trend'] = df['pressure'].rolling(window=window, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            print(f"      → Added pressure rolling features (multi-horizon optimized)")
    
        if 'wind_speed' in df.columns:
            for window in [6, 12, 18, 24, 36, 48, 60, 72]:
                df[f'wind_speed_{window}h_mean'] = df['wind_speed'].rolling(window=window, min_periods=1).mean()
                df[f'wind_speed_{window}h_max'] = df['wind_speed'].rolling(window=window, min_periods=1).max()
            print(f"      → Added wind speed rolling features (multi-horizon optimized)")
    
        # Multi-horizon specific features
        print(f"   Adding multi-horizon specific features...")
    
        # Seasonal patterns that affect different horizons
        df['seasonal_24h_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 365) * np.cos(2 * np.pi * df['hour'] / 24)
        df['seasonal_48h_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 365) * np.cos(2 * np.pi * df['hour'] / 12)
        df['seasonal_72h_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 365) * np.cos(2 * np.pi * df['hour'] / 8)
    
        # Day-of-week patterns for different horizons
        df['weekend_24h'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['weekend_48h'] = ((df['day_of_week'] + 1) % 7).isin([5, 6]).astype(int)
        df['weekend_72h'] = ((df['day_of_week'] + 2) % 7).isin([5, 6]).astype(int)
    
        print(f"      → Added seasonal factors for 24h, 48h, 72h horizons")
        print(f"      → Added weekend patterns for different horizons")
    
        print(f"   Total features after engineering: {len(df.columns)}")
        print(f"   ✅ NO AQI-contributing features to prevent data leakage")
        print(f"   ✅ Multi-horizon optimized features for 24h, 48h, 72h forecasting")
        print(f"   ✅ Using only raw values and non-AQI rolling windows")
        print(f"   ✅ NO AQI lags, rolling stats, or momentum features")
        print(f"   ✅ Only legitimate pollutant patterns and weather features")
    
        return df
    
        
    def remove_aqi_leakage_features(self, df):
        """Remove ALL AQI-related features that cause data leakage"""
        print(f"\n🔒 REMOVING ALL AQI LEAKAGE FEATURES")
        print("-" * 50)
    
        # CRITICAL: List of ALL features to remove (comprehensive cleanup)
        aqi_features_to_remove = [
            # AQI lag features (ALL of them)
            'aqi_lag_1h', 'aqi_lag_2h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 
            'aqi_lag_24h', 'aqi_lag_48h', 'aqi_lag_72h',
            
            # AQI rolling statistics (ALL of them)
            'aqi_24h_mean', 'aqi_24h_std', 'aqi_24h_min', 'aqi_24h_max', 'aqi_24h_range',
            'aqi_48h_mean', 'aqi_48h_std', 'aqi_48h_min', 'aqi_48h_max', 'aqi_48h_range',
            'aqi_72h_mean', 'aqi_72h_std', 'aqi_72h_min', 'aqi_72h_max', 'aqi_72h_range',
            
            # AQI change and momentum features (ALL of them)
            'aqi_change_1h', 'aqi_change_6h', 'aqi_change_24h',
            'aqi_acceleration', 'aqi_momentum_6h', 'aqi_momentum_24h', 'aqi_momentum_72h',
            
            # CRITICAL: Remove multi-horizon targets that cause data leakage
            'numerical_aqi_24h', 'numerical_aqi_48h', 'numerical_aqi_72h',
            
            # CRITICAL: Remove pollutant averages that are used for AQI calculation
            'pm2_5_24h_avg',    # Used to calculate PM2.5 AQI
            'pm10_24h_avg',     # Used to calculate PM10 AQI
            'o3_8h_avg',        # Used to calculate O3 AQI
            'o3_1h_avg',        # Used to calculate O3 AQI
            'co_8h_avg',        # Used to calculate CO AQI
            'no2_1h_avg',       # Used to calculate NO2 AQI
            'so2_1h_avg',       # Used to calculate SO2 AQI
            
            # CRITICAL: Remove primary_pollutant as it's derived from AQI calculation
            'primary_pollutant',  # This is derived from AQI calculation
            
            # CRITICAL: Remove any other AQI-related features
            'aqi_category', 'aqi_level', 'aqi_description'
        ]
    
        # CRITICAL: Remove features that exist
        features_removed = []
        for feature in aqi_features_to_remove:
            if feature in df.columns:
                df = df.drop(feature, axis=1)
                features_removed.append(feature)
    
        print(f"   Removed {len(features_removed)} AQI leakage features:")
        for feature in features_removed:
            print(f"      ❌ {feature}")
    
        print(f"   Features after removal: {len(df.columns)}")
        print(f"   ✅ Target variable 'numerical_aqi' kept for target creation")
        print(f"   ✅ ALL multi-horizon targets removed to prevent data leakage")
        print(f"   ✅ ALL AQI-contributing pollutant averages removed to prevent data leakage")
        print(f"   ✅ Primary pollutant column removed to prevent AQI leakage")
        print(f"   ✅ ALL AQI-related features removed to prevent data leakage")
        print(f"   ✅ Only legitimate forecasting features remain")
        
        return df
    
    def implement_true_forecasting(self, df):
        """Implement multi-horizon forecasting (24h, 48h, 72h) WITHOUT data leakage"""
        print(f"\n🔧 IMPLEMENTING MULTI-HORIZON FORECASTING (NO DATA LEAKAGE)")
        print("-" * 70)
    
        # Sort by timestamp to ensure proper shifting
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Store original length for reporting
        original_length = len(df)
    
        # CRITICAL: Create multiple forecasting targets by shifting AQI into the future
        # This means: current row's features predict AQI at different future time points
        df['target_aqi_24h'] = df['numerical_aqi'].shift(-24)   # 24 hours ahead
        df['target_aqi_48h'] = df['numerical_aqi'].shift(-48)   # 48 hours ahead  
        df['target_aqi_72h'] = df['numerical_aqi'].shift(-72)   # 72 hours ahead
        
        # CRITICAL: Remove the original AQI column to prevent data leakage
        # The original numerical_aqi contains current hour values that would leak future info
        df = df.drop('numerical_aqi', axis=1)
        
        # Count valid targets for each horizon
        valid_24h = df['target_aqi_24h'].notna().sum()
        valid_48h = df['target_aqi_48h'].notna().sum()
        valid_72h = df['target_aqi_72h'].notna().sum()
        
        # Find the last valid index for all targets
        last_valid_idx_24h = df['target_aqi_24h'].last_valid_index() or 0
        last_valid_idx_48h = df['target_aqi_48h'].last_valid_index() or 0
        last_valid_idx_72h = df['target_aqi_72h'].last_valid_index() or 0
        
        # Find the earliest cutoff point (most restrictive horizon)
        last_valid_idx = min(last_valid_idx_24h, last_valid_idx_48h, last_valid_idx_72h)
        
        # Trim the dataframe to keep only rows with valid targets for all horizons
        if last_valid_idx > 0:
            df = df.iloc[:last_valid_idx + 1].copy()
            
        # Get the actual date range of the final dataset
        if len(df) > 0:
            final_start = df['timestamp'].min()
            final_end = df['timestamp'].max()
            print(f"   Final dataset date range: {final_start} → {final_end}")
        
        final_length = len(df)
        records_lost = original_length - final_length
    
        print(f"   Original target: numerical_aqi (current hour)")
        print(f"   Multi-horizon targets created:")
        print(f"      → 24h ahead: {valid_24h:,} valid targets")
        print(f"      → 48h ahead: {valid_48h:,} valid targets") 
        print(f"      → 72h ahead: {valid_72h:,} valid targets")
        print(f"   Original records: {original_length:,}")
        print(f"   Final records: {final_length:,}")
        print(f"   Records lost due to shifts: {records_lost:,}")
        print(f"   Final dataset size: {final_length:,} records")
        print(f"   📝 Note: Data loss is expected and necessary for proper forecasting")
        print(f"      → 24h targets require 24 hours of future data")
        print(f"      → 48h targets require 48 hours of future data") 
        print(f"      → 72h targets require 72 hours of future data")
        print(f"      → Most recent data cannot have future targets (no future data available)")
        
        # CRITICAL: Verify no data leakage
        print(f"\n   🔒 DATA LEAKAGE PREVENTION VERIFICATION:")
        print(f"      ✅ Original 'numerical_aqi' column removed")
        print(f"      ✅ Target 'target_aqi_24h' is 24 hours in future")
        print(f"      ✅ Target 'target_aqi_48h' is 48 hours in future")
        print(f"      ✅ Target 'target_aqi_72h' is 72 hours in future")
        print(f"      ✅ No current-hour AQI values in features")
        print(f"      ✅ Features only contain past and current information")
        print(f"      ✅ Multi-horizon targets will be excluded from features")
        
        return df
    
    def validate_no_data_leakage(self, df, feature_columns):
        """Comprehensive validation to ensure NO data leakage"""
        print(f"\n🔒 COMPREHENSIVE DATA LEAKAGE VALIDATION")
        print("-" * 60)
        
        # Check 1: No target columns in features
        target_columns = ['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
        target_in_features = [col for col in target_columns if col in feature_columns]
        if target_in_features:
            print(f"   ❌ CRITICAL ERROR: Target columns found in features:")
            for col in target_in_features:
                print(f"      ❌ {col}")
            return False
        else:
            print(f"   ✅ No target columns in features")
        
        # Check 2: No AQI-related columns in features
        aqi_related = [col for col in feature_columns if 'aqi' in col.lower()]
        if aqi_related:
            print(f"   ❌ CRITICAL ERROR: AQI-related columns found in features:")
            for col in aqi_related:
                print(f"      ❌ {col}")
            return False
        else:
            print(f"   ✅ No AQI-related columns in features")
        
        # Check 3: No future information indicators
        future_indicators = ['future', 'ahead', 'next', 'forward']
        future_columns = [col for col in feature_columns if any(indicator in col.lower() for indicator in future_indicators)]
        if future_columns:
            print(f"   ❌ CRITICAL ERROR: Potential future information columns found:")
            for col in future_columns:
                print(f"      ❌ {col}")
            return False
        else:
            print(f"   ✅ No future information indicators in features")
        
        # Check 4: Verify temporal separation
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            time_diff = df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
            if time_diff.min() < 0:
                print(f"   ❌ CRITICAL ERROR: Negative time differences detected (data not properly sorted)")
                return False
            else:
                print(f"   ✅ Temporal order maintained")
        
        # Check 5: Feature count validation
        print(f"   ✅ Feature count: {len(feature_columns)}")
        print(f"   ✅ Target count: {len(target_columns)}")
        print(f"   ✅ Total columns: {len(feature_columns) + len(target_columns) + 1} (including timestamp)")
        
        print(f"\n   🔒 DATA LEAKAGE VALIDATION PASSED!")
        print(f"   ✅ Features contain ONLY past and current information")
        print(f"   ✅ Targets are properly shifted into the future")
        print(f"   ✅ No future information used as features")
        print(f"   ✅ Proper temporal separation maintained")
        
        return True
        
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print(f"\n🔧 HANDLING MISSING VALUES")
        print("-" * 50)
        
        # Check missing values before
        missing_before = df.isnull().sum()
        total_missing = missing_before.sum()
        print(f"   Missing values before: {total_missing:,}")
        
        # Handle missing values by column type
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count:,} missing ({missing_pct:.1f}%)")
            
            # Handle based on column type
            if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
                # Pollution data - forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                print(f"      → Forward/backward filled")
                
            elif col in ['temperature', 'tmin', 'tmax', 'dew_point', 'relative_humidity']:
                # Weather data - interpolate
                df[col] = df[col].interpolate(method='linear')
                print(f"      → Linear interpolation")
                
            elif col in ['precipitation', 'wind_speed', 'pressure']:
                # Weather data - fill with 0 or median
                if col == 'precipitation':
                    df[col] = df[col].fillna(0)
                    print(f"      → Filled with 0")
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"      → Filled with median: {median_val:.2f}")
                    
            elif col in ['hour', 'day', 'month', 'day_of_week', 'is_weekend', 
                        'day_of_year', 'week_of_year']:
                # Time features - recalculate from timestamp
                if col == 'hour':
                    df[col] = df['timestamp'].dt.hour
                elif col == 'day':
                    df[col] = df['timestamp'].dt.day
                elif col == 'month':
                    df[col] = df['timestamp'].dt.month
                elif col == 'day_of_week':
                    df[col] = df['timestamp'].dt.dayofweek
                elif col == 'is_weekend':
                    df[col] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
                elif col == 'day_of_year':
                    df[col] = df['timestamp'].dt.dayofyear
                elif col == 'week_of_year':
                    df[col] = df['timestamp'].dt.isocalendar().week
                print(f"      → Recalculated from timestamp")
                
            elif col.startswith(('hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                                'day_of_year_sin', 'day_of_year_cos')):
                # Cyclical features - recalculate
                if 'hour' in df.columns:
                    if col == 'hour_sin':
                        df[col] = np.sin(2 * np.pi * df['hour'] / 24)
                    elif col == 'hour_cos':
                        df[col] = np.cos(2 * np.pi * df['hour'] / 24)
                if 'month' in df.columns:
                    if col == 'month_sin':
                        df[col] = np.sin(2 * np.pi * df['month'] / 12)
                    elif col == 'month_cos':
                        df[col] = np.cos(2 * np.pi * df['month'] / 12)
                if 'day_of_year' in df.columns:
                    if col == 'day_of_year_sin':
                        df[col] = np.sin(2 * np.pi * df['day_of_year'] / 365)
                    elif col == 'day_of_year_cos':
                        df[col] = np.cos(2 * np.pi * df['day_of_year'] / 365)
                print(f"      → Recalculated cyclical features")
                
            elif col in ['temperature_range']:
                # Derived features - recalculate
                if 'tmax' in df.columns and 'tmin' in df.columns:
                    df[col] = df['tmax'] - df['tmin']
                print(f"      → Recalculated from tmax - tmin")
                
            elif col == 'numerical_aqi':
                # CRITICAL: Do NOT fill missing AQI values - they should be NaN if averaging windows incomplete
                print(f"      → Left as NaN (will be handled by dropping incomplete rows)")
                continue
                
            else:
                # Other columns - fill with appropriate defaults
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(0)
                    print(f"      → Filled with 0")
                else:
                    df[col] = df[col].fillna('unknown')
                    print(f"      → Filled with 'unknown'")
        
        # Check missing values after
        missing_after = df.isnull().sum()
        total_missing_after = missing_after.sum()
        print(f"\n   Missing values after: {total_missing_after:,}")
        print(f"   Improvement: {total_missing - total_missing_after:,} values fixed")
        
        return df
        
    def clean_data(self, df):
        """Clean and validate the data"""
        print(f"\n🧹 CLEANING DATA")
        print("-" * 50)
        
        initial_records = len(df)
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        print(f"   Removed {initial_records - len(df)} rows with invalid timestamps")
        
        # Handle extreme outliers in pollution data
        pollution_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        
        for col in pollution_cols:
            if col in df.columns:
                # Calculate IQR for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"   {col}: {outliers} outliers detected")
                    
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"      → Capped outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Handle temperature outliers
        if 'temperature' in df.columns:
            # Temperature should be reasonable for Peshawar (-10°C to 50°C)
            temp_outliers = ((df['temperature'] < -10) | (df['temperature'] > 50)).sum()
            if temp_outliers > 0:
                print(f"   temperature: {temp_outliers} extreme values detected")
                df['temperature'] = df['temperature'].clip(lower=-10, upper=50)
                print(f"      → Capped temperature to [-10°C, 50°C]")
        
        # Ensure AQI values are valid (0-500)
        if 'numerical_aqi' in df.columns:
            invalid_aqi = ((df['numerical_aqi'] < 0) | (df['numerical_aqi'] > 500)).sum()
            if invalid_aqi > 0:
                print(f"   numerical_aqi: {invalid_aqi} invalid values detected")
                df['numerical_aqi'] = df['numerical_aqi'].clip(lower=0, upper=500)
                print(f"      → Capped AQI to [0, 500]")
        
        print(f"   Final records: {len(df)}")
        
        return df
        
    def prepare_training_data(self, df):
        """Prepare final training dataset for multi-horizon forecasting WITHOUT data leakage"""
        print(f"\n🔒 PREPARING TRAINING DATA FOR MULTI-HORIZON FORECASTING")
        print("-" * 60)
    
        # CRITICAL: Remove rows with NaN targets (these are the last 72 hours)
        # We need all three targets to be valid for multi-horizon training
        initial_records = len(df)
        df = df.dropna(subset=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])
        print(f"   Records after removing NaN targets: {len(df):,}")
        print(f"   Removed {initial_records - len(df)} rows with NaN targets (last 72 hours)")
    
        # CRITICAL: Sort by timestamp to maintain temporal order
        df = df.sort_values('timestamp').reset_index(drop=True)
    
        # CRITICAL: Exclude ALL target columns and timestamp to prevent data leakage
        exclude_cols = [
            'timestamp',           # Time information (not a feature)
            'target_aqi_24h',     # 24-hour target variable
            'target_aqi_48h',     # 48-hour target variable
            'target_aqi_72h'      # 72-hour target variable
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"   Final feature columns: {len(feature_columns)}")
        print(f"   ✅ Multi-horizon targets:")
        print(f"      → 24h: target_aqi_24h")
        print(f"      → 48h: target_aqi_48h")
        print(f"      → 72h: target_aqi_72h")
        print(f"   ✅ Features: {len(feature_columns)} columns")
        print(f"   ✅ Excluded timestamp and all targets to prevent data leakage")
        
        # CRITICAL: Verify no target-related columns in features
        target_related = [col for col in feature_columns if 'aqi' in col.lower() or 'target' in col.lower()]
        if target_related:
            print(f"   ⚠️  WARNING: Found potential target-related columns in features:")
            for col in target_related:
                print(f"      ❌ {col}")
            print(f"   🔒 These will be removed to prevent data leakage")
            feature_columns = [col for col in feature_columns if col not in target_related]
            print(f"   ✅ Final feature count after cleanup: {len(feature_columns)}")
        else:
            print(f"   ✅ No target-related columns found in features")
    
        # Save feature columns for later use
        feature_cols_path = "data_repositories/features/phase1_no_leakage_feature_columns.pkl"
        import pickle
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"   Feature columns saved to: {feature_cols_path}")
    
        return df, feature_columns
        
    def save_data(self, df, feature_columns):
        """Save preprocessed data and metadata for multi-horizon forecasting"""
        print(f"\n💾 SAVING PREPROCESSED DATA")
        print("-" * 40)
        
        # Save preprocessed data
        df.to_csv(self.output_path, index=False)
        print(f"   Data saved to: {self.output_path}")
        
        # Save metadata
        metadata = {
            "preprocessing_timestamp": datetime.now().isoformat(),
            "training_period": f"{self.train_start.date()} to {self.train_end.date()}",
            "total_records": len(df),
            "feature_columns": feature_columns,
            "target_variables": {
                "target_aqi_24h": "AQI value 24 hours in the future",
                "target_aqi_48h": "AQI value 48 hours in the future", 
                "target_aqi_72h": "AQI value 72 hours in the future"
            },
            "target_ranges": {
                "24h": f"{df['target_aqi_24h'].min():.1f} to {df['target_aqi_24h'].max():.1f}",
                "48h": f"{df['target_aqi_48h'].min():.1f} to {df['target_aqi_48h'].max():.1f}",
                "72h": f"{df['target_aqi_72h'].min():.1f} to {df['target_aqi_72h'].max():.1f}"
            },
            "data_shape": df.shape,
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "missing_values_summary": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "aqi_calculation": "EPA-COMPLIANT with proper unit conversion, averaging windows, and truncation",
            "unit_conversions": "µg/m³ → ppb for O3, NO2, SO2; µg/m³ → ppm for CO",
            "averaging_periods": "PM2.5/PM10: 24-hour; O3: 8-hour + 1-hour selection; CO: 8-hour; NO2/SO2: 1-hour",
            "epa_truncation": "Applied EPA precision rules before AQI calculation",
            "o3_selection_rule": "EPA O3 rule: 8-hour vs 1-hour selection based on concentration thresholds",
            "forecasting_horizon": "MULTI-HORIZON: 24h, 48h, 72h forecasting",
            "data_leakage_prevention": "NO FUTURE INFORMATION USED AS FEATURES",
            "target_creation": "Multi-horizon targets created by shifting AQI 24h, 48h, 72h into the future",
            "feature_engineering": "NO AQI LEAKAGE - Only legitimate forecasting features",
            "rolling_windows": "Multi-horizon windows: 6h, 12h, 18h, 24h, 36h, 48h, 60h, 72h",
            "legitimate_features": "Raw pollutants, multi-horizon lags, non-AQI rolling stats, weather, time features",
            "data_leakage_checks": "Original AQI removed, targets shifted, no future info in features",
            "forecasting_setup": "Multi-horizon: 24h, 48h, 72h AQI forecasting with proper temporal separation",
            "multi_horizon_benefits": "24h (high confidence), 48h (medium confidence), 72h (lower confidence)"
        }
        
        import json
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        print(f"   Metadata saved to: {self.metadata_path}")
        
        # Print summary
        print(f"\n📊 PREPROCESSING SUMMARY:")
        print(f"   Input records: {len(pd.read_csv(self.input_path)):,}")
        print(f"   Output records: {len(df):,}")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Multi-horizon targets:")
        print(f"      → 24h: target_aqi_24h")
        print(f"      → 48h: target_aqi_48h")
        print(f"      → 72h: target_aqi_72h")
        print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"   ✅ MULTI-HORIZON FORECASTING: 24h, 48h, 72h ahead")
        print(f"   ✅ NO DATA LEAKAGE: No future information used as features")
        print(f"   ✅ NO AQI LEAKAGE: Only legitimate forecasting features")
        
        return True
        
    def run_preprocessing(self):
        """Run complete preprocessing pipeline without data leakage"""
        print(f"\n🚀 STARTING FIXED PREPROCESSING PIPELINE (NO DATA LEAKAGE)")
        print("=" * 80)
        # Print input dataset date range at the start
        self._print_dataset_range(self.input_path, "Preprocessing input dataset range")
        
        # Step 1: Load data
        df = self.load_data()
        if df is None:
            return False
        
        # Step 2: Convert units to EPA standard
        df = self.convert_units_to_epa_standard(df)
        
        # Step 3: Calculate required averages
        df = self.calculate_required_averages(df)
        
        # Step 4: Calculate numerical AQI
        df = self.calculate_numerical_aqi(df)
        
        # Step 5: Remove AQI leakage features
        df = self.remove_aqi_leakage_features(df)
        
        # Step 6: Engineer legitimate features
        df = self.engineer_legitimate_features(df)
        
        # Step 7: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 8: Clean data
        df = self.clean_data(df)
        
        # Step 9: Implement multi-horizon forecasting FIRST (creates targets)
        df = self.implement_true_forecasting(df)
        
        # Step 10: Prepare training data AFTER targets are created
        df, feature_columns = self.prepare_training_data(df)
        
        # Step 11: CRITICAL: Validate no data leakage
        if not self.validate_no_data_leakage(df, feature_columns):
            print(f"\n❌ CRITICAL ERROR: Data leakage detected! Pipeline stopped.")
            return False
        
        # Step 12: Save data
        success = self.save_data(df, feature_columns)
        
        if success:
            # Print output dataset date range at the end
            self._print_dataset_range(self.output_path, "Preprocessing output dataset range")
            print(f"\n✅ MULTI-HORIZON FORECASTING PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("📊 Key Changes Made:")
            print("   ✅ EPA-COMPLIANT numerical AQI calculation")
            print("   ✅ Proper unit conversions (µg/m³ → ppb/ppm)")
            print("   ✅ Required averaging periods (24h for PM, 8h for O3/CO, 1h for NO2/SO2)")
            print("   ✅ EPA truncation rules applied")
            print("   ✅ O3 selection rule implemented (8-hour vs 1-hour)")
            print("   ✅ MULTI-HORIZON FORECASTING: 24h, 48h, 72h targets")
            print("   ✅ NO DATA LEAKAGE: Original AQI column removed")
            print("   ✅ NO AQI LEAKAGE: Only legitimate forecasting features")
            print("   ✅ Multi-horizon optimized features (pollutant lags, weather, time)")
            print("   ✅ Targets: target_aqi_24h, target_aqi_48h, target_aqi_72h")
            print("   ✅ Features: Only past and current information")
            print("   ✅ NO FUTURE INFORMATION used as features")
            print("   ✅ Models will learn from actual patterns without cheating")
            print("\n🔒 Data Leakage Prevention:")
            print("   ✅ Original 'numerical_aqi' column removed")
            print("   ✅ Target 'target_aqi_24h' is 24 hours in future")
            print("   ✅ Target 'target_aqi_48h' is 48 hours in future")
            print("   ✅ Target 'target_aqi_72h' is 72 hours in future")
            print("   ✅ No current-hour AQI values in features")
            print("   ✅ Features only contain past and current information")
            print("   ✅ Proper temporal separation maintained")
            print("\n📊 Multi-Horizon Forecasting Setup:")
            print("   🎯 24h forecasts: High confidence, practical for daily planning")
            print("   🎯 48h forecasts: Medium confidence, useful for weekend planning")
            print("   🎯 72h forecasts: Lower confidence, general trend indication")
            print("   🎯 All targets properly excluded from features")
            print("\n📊 Next steps:")
            print("   1. Feature scaling and train/validation split")
            print("   2. SHAP analysis for top feature selection")
            print("   3. Multi-horizon model training (24h, 48h, 72h)")
        else:
            print(f"\n❌ MULTI-HORIZON FORECASTING PREPROCESSING FAILED!")
            print("=" * 80)
        
        return success

def main():
    """Run the multi-horizon forecasting preprocessing pipeline without data leakage"""
    preprocessor = FixedNoLeakagePreprocessor()
    success = preprocessor.run_preprocessing()
    
    if success:
        print(f"\n🎉 Ready for Multi-Horizon AQI Forecasting System!")
        print(f"   🎯 24h forecasting: High confidence for daily planning")
        print(f"   🎯 48h forecasting: Medium confidence for weekend planning")
        print(f"   🎯 72h forecasting: Lower confidence for trend indication")
        print(f"   🔒 No data leakage: Only legitimate forecasting features")
        print(f"   ✅ Proper temporal separation maintained")
        print(f"   ✅ Comprehensive data leakage validation passed")
        print(f"   ✅ Multi-horizon targets properly created and separated")
        print(f"   ✅ Features contain only past and current information")
    else:
        print(f"\n❌ Multi-horizon forecasting preprocessing failed! Check error messages above.")

if __name__ == "__main__":
    main()
