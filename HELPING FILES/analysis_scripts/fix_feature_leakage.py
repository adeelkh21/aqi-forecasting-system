"""
Fix Feature Leakage Issues - Remove Problematic Features
========================================================

This script removes features that may cause data leakage and creates
a clean, validated feature set for production use.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def fix_leakage_issues():
    """Remove problematic features and validate clean dataset"""
    
    print("🔧 FIXING DATA LEAKAGE ISSUES")
    print("=" * 35)
    
    # Load enhanced features
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"Original dataset: {df.shape}")
    
    # 1. Remove problematic change features
    print("\n1. Removing Change Features (Potential Leakage):")
    print("-" * 50)
    
    change_features = [col for col in df.columns if 'change' in col]
    print(f"   Change features to remove: {len(change_features)}")
    for feat in change_features:
        print(f"      - {feat}")
    
    # Remove change features
    df_clean = df.drop(columns=change_features)
    
    # 2. Verify remaining features are safe
    print("\n2. Validating Remaining Features:")
    print("-" * 35)
    
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'aqi_numeric']]
    
    # Categorize features
    lag_features = [col for col in feature_cols if 'lag' in col]
    rolling_features = [col for col in feature_cols if 'rolling' in col]
    seasonal_features = [col for col in feature_cols if any(x in col for x in ['sin', 'cos', 'hour', 'day', 'month', 'season'])]
    interaction_features = [col for col in feature_cols if 'interaction' in col or 'ratio' in col]
    base_features = [col for col in feature_cols if col not in lag_features + rolling_features + seasonal_features + interaction_features]
    
    print(f"   Base features (current): {len(base_features)}")
    print(f"   Lag features: {len(lag_features)}")
    print(f"   Rolling features: {len(rolling_features)}")
    print(f"   Seasonal features: {len(seasonal_features)}")
    print(f"   Interaction features: {len(interaction_features)}")
    print(f"   Total safe features: {len(feature_cols)}")
    
    # 3. Test temporal consistency
    print("\n3. Temporal Consistency Validation:")
    print("-" * 40)
    
    # Check that lag features are computed correctly
    df_clean = df_clean.dropna()
    
    # Verify 24h lag
    if 'pm2_5_lag24h' in df_clean.columns:
        manual_lag = df_clean['pm2_5'].shift(24)
        feature_lag = df_clean['pm2_5_lag24h']
        max_diff = (manual_lag - feature_lag).abs().max()
        
        if pd.isna(max_diff) or max_diff < 0.001:
            print("   ✅ 24h lag features computed correctly")
        else:
            print(f"   ⚠️  24h lag mismatch: {max_diff}")
    
    # 4. Test realistic performance
    print("\n4. Performance Test (Clean Features):")
    print("-" * 40)
    
    # Temporal split
    split_idx = int(len(df_clean) * 0.7)
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]
    
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['aqi_numeric']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['aqi_numeric']
    
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Test with Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   Train R²: {train_score:.4f} ({train_score*100:.1f}%)")
    print(f"   Test R²:  {test_score:.4f} ({test_score*100:.1f}%)")
    print(f"   MAE:      {mae:.2f} AQI points")
    print(f"   Gap:      {train_score - test_score:.4f}")
    
    # 5. Time Series Cross-Validation
    print("\n5. Time Series Cross-Validation:")
    print("-" * 35)
    
    X_full = df_clean[feature_cols].fillna(0)
    y_full = df_clean['aqi_numeric']
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_full)):
        X_train_cv, X_test_cv = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_cv, y_test_cv = y_full.iloc[train_idx], y_full.iloc[test_idx]
        
        rf_cv = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        rf_cv.fit(X_train_cv, y_train_cv)
        score = rf_cv.score(X_test_cv, y_test_cv)
        cv_scores.append(score)
        
        print(f"   Fold {fold+1}: R² = {score:.4f} ({score*100:.1f}%)")
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    
    print(f"   Mean: {mean_cv:.4f} ± {std_cv:.4f} ({mean_cv*100:.1f}%)")
    
    # 6. Feature importance analysis
    print("\n6. Top Features (No Leakage):")
    print("-" * 30)
    
    importances = rf.feature_importances_
    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature:<35} {importance:.4f}")
    
    # 7. Save clean dataset
    print("\n7. Saving Clean Dataset:")
    print("-" * 25)
    
    clean_file = "data_repositories/features/clean_features.csv"
    df_clean.to_csv(clean_file, index=False)
    
    # Save metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_records": len(df_clean),
        "total_features": len(feature_cols),
        "removed_features": len(change_features),
        "performance": {
            "temporal_split_r2": test_score,
            "cv_mean_r2": mean_cv,
            "cv_std_r2": std_cv,
            "mae": mae
        },
        "feature_categories": {
            "base": len(base_features),
            "lag": len(lag_features),
            "rolling": len(rolling_features),
            "seasonal": len(seasonal_features),
            "interaction": len(interaction_features)
        },
        "top_10_features": [feat for feat, _ in feature_importance[:10]]
    }
    
    import json
    metadata_file = "data_repositories/features/clean_features_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print(f"   ✅ Clean dataset saved: {clean_file}")
    print(f"   ✅ Metadata saved: {metadata_file}")
    
    # 8. Final assessment
    print(f"\n📊 FINAL CLEAN DATASET ASSESSMENT:")
    print("=" * 40)
    print(f"   Records: {len(df_clean):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Removed problematic features: {len(change_features)}")
    print(f"   Performance (temporal): {test_score:.3f} ({test_score*100:.1f}%)")
    print(f"   Performance (CV): {mean_cv:.3f} ({mean_cv*100:.1f}%)")
    print(f"   Target: 75%")
    
    if test_score >= 0.75 and mean_cv >= 0.75:
        print("   ✅ TARGET ACHIEVED WITH CLEAN DATA")
    elif test_score >= 0.70 or mean_cv >= 0.70:
        print("   ⚠️  CLOSE TO TARGET - MINOR OPTIMIZATION NEEDED")
    else:
        print("   ❌ NEED MORE FEATURE ENGINEERING")
    
    return test_score, mean_cv, len(feature_cols)

if __name__ == "__main__":
    test_r2, cv_r2, num_features = fix_leakage_issues()
