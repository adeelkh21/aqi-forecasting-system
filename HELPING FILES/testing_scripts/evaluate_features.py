"""
Feature Evaluation Script - Assess Phase 2 Readiness for 75% Target
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def evaluate_current_features():
    """Evaluate current feature set for 3-day forecasting capability"""
    
    print("🔍 PHASE 2 EVALUATION: Readiness for 75% Target & 3-Day Forecasting")
    print("=" * 70)
    
    # Load engineered features
    df = pd.read_csv('data_repositories/features/engineered_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi_numeric']]
    X = df[feature_cols].fillna(0)
    y = df['aqi_numeric']
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   AQI range: {y.min():.1f} - {y.max():.1f}")
    
    # 1. Current Model Performance (Random Forest)
    print(f"\n🎯 Current Model Performance (Baseline):")
    print("-" * 45)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"   R² Score: {r2:.3f} ({r2*100:.1f}%)")
    print(f"   MAE: {mae:.2f} AQI points")
    print(f"   RMSE: {rmse:.2f} AQI points")
    
    # 2. Time Series Performance (More realistic for forecasting)
    print(f"\n⏰ Time Series Validation (3-Fold):")
    print("-" * 40)
    
    tscv = TimeSeriesSplit(n_splits=3)
    r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_ts, X_test_ts = X.iloc[train_idx], X.iloc[test_idx]
        y_train_ts, y_test_ts = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_ts = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_ts.fit(X_train_ts, y_train_ts)
        y_pred_ts = rf_ts.predict(X_test_ts)
        
        fold_r2 = r2_score(y_test_ts, y_pred_ts)
        r2_scores.append(fold_r2)
        print(f"   Fold {fold+1}: R² = {fold_r2:.3f} ({fold_r2*100:.1f}%)")
    
    avg_r2 = np.mean(r2_scores)
    print(f"   Average: R² = {avg_r2:.3f} ({avg_r2*100:.1f}%)")
    
    # 3. Feature Analysis
    print(f"\n📋 Feature Quality Analysis:")
    print("-" * 35)
    
    # Load feature importance
    importance_df = pd.read_csv('data_repositories/feature_analysis/feature_importance.csv')
    
    top_features = importance_df.head(10)
    print(f"   Top features contributing: {len(top_features)}")
    print(f"   Highest importance: {top_features.iloc[0]['feature']} ({top_features.iloc[0]['combined_score']:.2f})")
    
    # Feature diversity
    weather_features = [f for f in feature_cols if any(w in f for w in ['temperature', 'humidity', 'wind', 'pressure'])]
    pollution_features = [f for f in feature_cols if any(p in f for p in ['pm2_5', 'pm10', 'no2', 'o3'])]
    time_features = [f for f in feature_cols if any(t in f for t in ['hour', 'day', 'weekend'])]
    lag_features = [f for f in feature_cols if 'lag' in f]
    
    print(f"   Weather features: {len(weather_features)}")
    print(f"   Pollution features: {len(pollution_features)}")
    print(f"   Time features: {len(time_features)}")
    print(f"   Lag features: {len(lag_features)}")
    
    # 4. Gap Analysis for 75% Target
    print(f"\n📈 Target Analysis:")
    print("-" * 25)
    print(f"   Current performance: {avg_r2*100:.1f}%")
    print(f"   Target performance: 75.0%")
    gap = 75 - (avg_r2*100)
    print(f"   Performance gap: {gap:.1f} percentage points")
    
    if gap > 0:
        print(f"   Status: ❌ Need {gap:.1f}% improvement")
    else:
        print(f"   Status: ✅ Target achieved!")
    
    # 5. 3-Day Forecasting Assessment
    print(f"\n🔮 3-Day Forecasting Readiness:")
    print("-" * 35)
    
    max_lag = 24  # Maximum lag we have is 24 hours
    forecasting_horizon = 72  # 3 days = 72 hours
    
    print(f"   Current max lag: {max_lag} hours")
    print(f"   Target horizon: {forecasting_horizon} hours")
    print(f"   Lag coverage: {max_lag/forecasting_horizon*100:.1f}%")
    
    if max_lag < forecasting_horizon:
        print(f"   Status: ⚠️  Limited - need longer lags for 3-day forecasting")
    else:
        print(f"   Status: ✅ Adequate lag coverage")
    
    # 6. Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 20)
    
    if gap > 20:
        print("   🚨 CRITICAL: Need significant feature engineering")
        print("   - Add more complex features (rolling means, seasonality)")
        print("   - Consider external data sources")
        print("   - Implement advanced models (XGBoost, Neural Networks)")
    elif gap > 10:
        print("   ⚠️  MODERATE: Need feature improvements")
        print("   - Add rolling statistics features")
        print("   - Implement better lag strategies")
        print("   - Tune hyperparameters")
    elif gap > 5:
        print("   ✅ MINOR: Small optimizations needed")
        print("   - Hyperparameter tuning")
        print("   - Model ensemble")
    else:
        print("   🎉 EXCELLENT: Ready for target!")
    
    # 3-day forecasting recommendations
    if max_lag < forecasting_horizon:
        print(f"\n   📅 For 3-day forecasting:")
        print(f"   - Add 48h and 72h lag features")
        print(f"   - Implement multi-step forecasting strategy")
        print(f"   - Consider recursive forecasting approach")
    
    return avg_r2, gap

if __name__ == "__main__":
    avg_r2, gap = evaluate_current_features()
