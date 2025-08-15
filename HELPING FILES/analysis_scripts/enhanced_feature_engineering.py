"""
Enhanced Feature Engineering for 75% Target & 3-Day Forecasting
================================================================

This script creates additional features needed to achieve 75% accuracy
and enable reliable 3-day AQI forecasting.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features():
    """Create enhanced features for better performance"""
    
    print("🚀 Enhanced Feature Engineering for 75% Target")
    print("=" * 55)
    
    # Load base features
    df = pd.read_csv('data_repositories/features/engineered_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"📊 Starting with {len(df)} records and {len(df.columns)-2} features")
    
    # 1. Extended Lag Features (48h, 72h for 3-day forecasting)
    print("\n⏳ Creating Extended Lag Features (48h, 72h)...")
    
    extended_lag_features = []
    
    # 48h and 72h lags for key pollutants
    for hours in [48, 72]:
        for col in ['pm2_5', 'pm10', 'no2', 'o3', 'aqi_numeric']:
            if col in df.columns:
                lag_col = f"{col}_lag{hours}h"
                df[lag_col] = df[col].shift(hours)
                extended_lag_features.append(lag_col)
                print(f"   ✅ {lag_col}")
    
    # 48h and 72h lags for key weather
    for hours in [48, 72]:
        for col in ['temperature', 'relative_humidity', 'pressure']:
            if col in df.columns:
                lag_col = f"{col}_lag{hours}h"
                df[lag_col] = df[col].shift(hours)
                extended_lag_features.append(lag_col)
                print(f"   ✅ {lag_col}")
    
    # 2. Rolling Statistics (Moving Averages)
    print("\n📊 Creating Rolling Statistics Features...")
    
    rolling_features = []
    
    # Rolling means for different windows
    for window in [6, 12, 24]:  # 6h, 12h, 24h windows
        for col in ['pm2_5', 'pm10', 'aqi_numeric', 'temperature', 'relative_humidity']:
            if col in df.columns:
                roll_col = f"{col}_rolling_mean_{window}h"
                df[roll_col] = df[col].rolling(window=window, min_periods=1).mean()
                rolling_features.append(roll_col)
                print(f"   ✅ {roll_col}")
    
    # Rolling standard deviations (volatility)
    for window in [12, 24]:
        for col in ['pm2_5', 'pm10', 'aqi_numeric']:
            if col in df.columns:
                roll_col = f"{col}_rolling_std_{window}h"
                df[roll_col] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                rolling_features.append(roll_col)
                print(f"   ✅ {roll_col}")
    
    # 3. Seasonal and Cyclical Features
    print("\n🌀 Creating Seasonal/Cyclical Features...")
    
    seasonal_features = []
    
    # Hour cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    seasonal_features.extend(['hour_sin', 'hour_cos'])
    
    # Day of week cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    seasonal_features.extend(['dow_sin', 'dow_cos'])
    
    # Month/Season
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    seasonal_features.extend(['month', 'season'])
    
    # Day of year cyclical
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    seasonal_features.extend(['day_of_year', 'doy_sin', 'doy_cos'])
    
    for feat in seasonal_features:
        print(f"   ✅ {feat}")
    
    # 4. Interaction Features
    print("\n🔗 Creating Interaction Features...")
    
    interaction_features = []
    
    # Weather-pollution interactions
    if 'pm2_5' in df.columns and 'relative_humidity' in df.columns:
        df['pm25_humidity_interaction'] = df['pm2_5'] * df['relative_humidity']
        interaction_features.append('pm25_humidity_interaction')
    
    if 'pm2_5' in df.columns and 'temperature' in df.columns:
        df['pm25_temp_interaction'] = df['pm2_5'] * df['temperature']
        interaction_features.append('pm25_temp_interaction')
    
    if 'pm10' in df.columns and 'wind_speed' in df.columns:
        df['pm10_wind_interaction'] = df['pm10'] / (df['wind_speed'] + 0.1)  # Wind disperses particles
        interaction_features.append('pm10_wind_interaction')
    
    # Pollutant ratios
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.1)
        interaction_features.append('pm25_pm10_ratio')
    
    for feat in interaction_features:
        print(f"   ✅ {feat}")
    
    # 5. Trend Features
    print("\n📈 Creating Trend Features...")
    
    trend_features = []
    
    # Rate of change features
    for col in ['pm2_5', 'pm10', 'aqi_numeric', 'temperature']:
        if col in df.columns:
            # 1-hour change
            change_col = f"{col}_change_1h"
            df[change_col] = df[col].diff(1).fillna(0)
            trend_features.append(change_col)
            
            # 3-hour change
            change_col = f"{col}_change_3h"
            df[change_col] = df[col].diff(3).fillna(0)
            trend_features.append(change_col)
            print(f"   ✅ {col} change features")
    
    # 6. Peak/Valley Detection
    print("\n🏔️ Creating Peak/Valley Features...")
    
    peak_features = []
    
    # Daily min/max features
    df['hour_since_midnight'] = df['hour']
    
    # Peak pollution hours (typically morning/evening)
    df['is_peak_pollution_hour'] = df['hour'].isin([7, 8, 9, 18, 19, 20]).astype(int)
    peak_features.append('is_peak_pollution_hour')
    
    # Low pollution hours (typically afternoon)
    df['is_low_pollution_hour'] = df['hour'].isin([13, 14, 15, 16]).astype(int)
    peak_features.append('is_low_pollution_hour')
    
    for feat in peak_features:
        print(f"   ✅ {feat}")
    
    # Summary
    total_new_features = len(extended_lag_features) + len(rolling_features) + len(seasonal_features) + len(interaction_features) + len(trend_features) + len(peak_features)
    
    print(f"\n📋 Enhanced Feature Summary:")
    print(f"   Extended lag features: {len(extended_lag_features)}")
    print(f"   Rolling statistics: {len(rolling_features)}")
    print(f"   Seasonal/cyclical: {len(seasonal_features)}")
    print(f"   Interaction features: {len(interaction_features)}")
    print(f"   Trend features: {len(trend_features)}")
    print(f"   Peak/valley features: {len(peak_features)}")
    print(f"   Total new features: {total_new_features}")
    print(f"   Total features now: {len(df.columns)-2}")
    
    # Save enhanced features
    output_file = "data_repositories/features/enhanced_features.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Enhanced features saved to: {output_file}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Ready for advanced modeling to achieve 75% target!")
    
    return df

if __name__ == "__main__":
    enhanced_df = create_enhanced_features()
