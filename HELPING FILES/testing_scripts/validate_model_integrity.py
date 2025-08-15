"""
Model Integrity Validation - Check for Overfitting, Underfitting, Data Leakage
=============================================================================

This script performs comprehensive validation to detect:
- Data leakage (future information bleeding into past)
- Overfitting (too complex model memorizing noise)
- Underfitting (too simple model missing patterns)
- Feature contamination (target information in features)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, validation_curve, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def check_data_leakage():
    """Check for data leakage issues"""
    print("🔍 CHECKING FOR DATA LEAKAGE")
    print("=" * 40)
    
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 1. Check for future information in lag features
    print("\n1. Temporal Consistency Check:")
    print("-" * 30)
    
    leakage_issues = []
    
    # Check if any feature contains future information
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'aqi_numeric']]
    
    # Check change features - these might be problematic
    change_features = [col for col in feature_cols if 'change' in col]
    print(f"   Change features found: {len(change_features)}")
    
    if len(change_features) > 0:
        print("   ⚠️  Change features detected - potential leakage risk")
        for feat in change_features[:5]:
            print(f"      - {feat}")
        leakage_issues.append("Change features may contain forward-looking information")
    
    # 2. Check correlation between target and features
    print("\n2. Target-Feature Correlation Analysis:")
    print("-" * 40)
    
    df_clean = df.dropna()
    correlations = df_clean[feature_cols + ['aqi_numeric']].corr()['aqi_numeric'].abs().sort_values(ascending=False)
    
    highly_correlated = correlations[correlations > 0.95]
    print(f"   Features with >95% correlation to target: {len(highly_correlated)-1}")
    
    if len(highly_correlated) > 1:  # -1 because target correlates with itself
        print("   🚨 POTENTIAL LEAKAGE - Suspiciously high correlations:")
        for feat, corr in highly_correlated.head(10).items():
            if feat != 'aqi_numeric':
                print(f"      {feat}: {corr:.4f}")
        leakage_issues.append("Extremely high correlations suggest leakage")
    
    # 3. Check rolling features computation
    print("\n3. Rolling Features Validation:")
    print("-" * 35)
    
    rolling_features = [col for col in feature_cols if 'rolling' in col]
    print(f"   Rolling features: {len(rolling_features)}")
    
    # Sample check: verify rolling mean is computed correctly
    if 'aqi_numeric_rolling_mean_6h' in df.columns:
        # Check first few values
        manual_rolling = df['aqi_numeric'].rolling(window=6, min_periods=1).mean()
        feature_rolling = df['aqi_numeric_rolling_mean_6h']
        
        diff = (manual_rolling - feature_rolling).abs().max()
        if diff > 0.001:
            print(f"   ⚠️  Rolling computation mismatch: max diff = {diff:.6f}")
            leakage_issues.append("Rolling features may be computed incorrectly")
        else:
            print("   ✅ Rolling features computed correctly")
    
    return leakage_issues

def check_overfitting():
    """Check for overfitting using multiple validation techniques"""
    print("\n🎯 CHECKING FOR OVERFITTING")
    print("=" * 35)
    
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df_clean = df.dropna()
    
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'aqi_numeric']]
    X = df_clean[feature_cols].fillna(0)
    y = df_clean['aqi_numeric']
    
    overfitting_issues = []
    
    # 1. Train vs Validation Performance
    print("\n1. Train vs Validation Performance:")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different model complexities
    models = {
        'Simple RF (10 trees)': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
        'Medium RF (100 trees)': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Complex RF (300 trees)': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
        'Very Complex RF (500 trees)': RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        gap = train_score - test_score
        
        print(f"   {name}:")
        print(f"      Train R²: {train_score:.4f}")
        print(f"      Test R²:  {test_score:.4f}")
        print(f"      Gap:      {gap:.4f}")
        
        if gap > 0.1:
            print(f"      ⚠️  Large train-test gap suggests overfitting")
            overfitting_issues.append(f"{name} shows overfitting (gap: {gap:.3f})")
        elif gap < 0.05:
            print(f"      ✅ Good generalization")
        print()
    
    # 2. Time Series Cross-Validation (More Realistic)
    print("2. Time Series Cross-Validation:")
    print("-" * 35)
    
    tscv = TimeSeriesSplit(n_splits=5)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_ts, X_test_ts = X.iloc[train_idx], X.iloc[test_idx]
        y_train_ts, y_test_ts = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_model.fit(X_train_ts, y_train_ts)
        score = rf_model.score(X_test_ts, y_test_ts)
        fold_scores.append(score)
        print(f"   Fold {fold+1}: R² = {score:.4f}")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"   Mean: {mean_score:.4f} ± {std_score:.4f}")
    
    if std_score > 0.2:
        print("   ⚠️  High variance across folds - potential overfitting")
        overfitting_issues.append(f"High CV variance: {std_score:.3f}")
    else:
        print("   ✅ Stable performance across folds")
    
    # 3. Learning Curves
    print("\n3. Learning Curve Analysis:")
    print("-" * 30)
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        X, y, train_sizes=train_sizes, cv=3, random_state=42, n_jobs=-1
    )
    
    final_train_score = np.mean(train_scores[-1])
    final_val_score = np.mean(val_scores[-1])
    final_gap = final_train_score - final_val_score
    
    print(f"   Final training score: {final_train_score:.4f}")
    print(f"   Final validation score: {final_val_score:.4f}")
    print(f"   Final gap: {final_gap:.4f}")
    
    if final_gap > 0.15:
        print("   ⚠️  Large learning curve gap - overfitting likely")
        overfitting_issues.append(f"Learning curve gap: {final_gap:.3f}")
    
    return overfitting_issues

def check_feature_contamination():
    """Check for feature contamination (target info in features)"""
    print("\n🧹 CHECKING FOR FEATURE CONTAMINATION")
    print("=" * 45)
    
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df_clean = df.dropna()
    
    contamination_issues = []
    
    # 1. Check for features that are mathematical transformations of target
    print("1. Mathematical Transformation Check:")
    print("-" * 40)
    
    target = df_clean['aqi_numeric']
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'aqi_numeric']]
    
    # Check for features that are just scaled/shifted versions of target
    suspicious_features = []
    for col in feature_cols:
        if 'aqi' in col and 'lag' not in col and 'rolling' not in col and 'change' not in col:
            feature = df_clean[col]
            
            # Check if feature is linearly related to target (excluding lag features)
            if len(feature.unique()) > 10:  # Only check numeric features
                correlation = np.corrcoef(target, feature)[0, 1]
                if abs(correlation) > 0.99:
                    suspicious_features.append((col, correlation))
    
    if suspicious_features:
        print("   🚨 HIGHLY SUSPICIOUS FEATURES:")
        for feat, corr in suspicious_features:
            print(f"      {feat}: correlation = {corr:.6f}")
        contamination_issues.extend([f"Suspicious feature: {feat}" for feat, _ in suspicious_features])
    else:
        print("   ✅ No obvious mathematical transformations found")
    
    # 2. Check temporal ordering
    print("\n2. Temporal Ordering Validation:")
    print("-" * 35)
    
    df_clean = df_clean.sort_values('timestamp')
    
    # Check if any feature uses future information
    change_features = [col for col in feature_cols if 'change' in col]
    
    if change_features:
        print(f"   Found {len(change_features)} change features")
        
        # Verify change features don't use future data
        for feat in change_features[:3]:  # Check first 3
            if feat in df_clean.columns:
                # Check if changes are computed forward or backward
                actual_changes = df_clean[feat].dropna()
                if len(actual_changes) > 100:
                    # If most changes are exactly 0, it might indicate wrong computation
                    zero_rate = (actual_changes == 0).mean()
                    if zero_rate > 0.5:
                        print(f"   ⚠️  {feat}: {zero_rate:.1%} zero values - potential issue")
                        contamination_issues.append(f"Suspicious change feature: {feat}")
    
    return contamination_issues

def realistic_performance_test():
    """Test model performance with proper temporal splits"""
    print("\n📊 REALISTIC PERFORMANCE TEST")
    print("=" * 35)
    
    df = pd.read_csv('data_repositories/features/enhanced_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df_clean = df.dropna()
    
    print(f"Clean dataset: {len(df_clean)} records")
    
    # 1. Temporal split (train on first 70%, test on last 30%)
    split_idx = int(len(df_clean) * 0.7)
    
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]
    
    print(f"Training period: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
    print(f"Testing period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
    
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'aqi_numeric']]
    
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['aqi_numeric']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['aqi_numeric']
    
    # 2. Test different models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Simple RF': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
        'Medium RF': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mae': mae,
            'rmse': rmse,
            'gap': train_score - test_score
        }
        
        print(f"\n{name}:")
        print(f"   Train R²: {train_score:.4f}")
        print(f"   Test R²:  {test_score:.4f}")
        print(f"   MAE:      {mae:.2f}")
        print(f"   RMSE:     {rmse:.2f}")
        print(f"   Gap:      {train_score - test_score:.4f}")
    
    # 3. Find best realistic performance
    best_test_r2 = max([r['test_r2'] for r in results.values()])
    
    print(f"\n📈 REALISTIC PERFORMANCE SUMMARY:")
    print(f"   Best Test R²: {best_test_r2:.4f} ({best_test_r2*100:.1f}%)")
    print(f"   Target: 75%")
    
    if best_test_r2 >= 0.75:
        print("   ✅ Target achieved with proper validation")
    elif best_test_r2 >= 0.65:
        print("   ⚠️  Close to target - need optimization")
    else:
        print("   ❌ Significant gap to target")
    
    return best_test_r2

def main():
    """Run complete validation suite"""
    print("🔍 MODEL INTEGRITY VALIDATION SUITE")
    print("=" * 50)
    
    # Run all checks
    leakage_issues = check_data_leakage()
    overfitting_issues = check_overfitting()
    contamination_issues = check_feature_contamination()
    realistic_r2 = realistic_performance_test()
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 25)
    
    all_issues = leakage_issues + overfitting_issues + contamination_issues
    
    if all_issues:
        print("🚨 ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No major issues detected")
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Realistic Performance: {realistic_r2:.3f} ({realistic_r2*100:.1f}%)")
    print(f"   Issues Found: {len(all_issues)}")
    
    if realistic_r2 >= 0.75 and len(all_issues) == 0:
        print("   ✅ MODEL IS VALID AND READY")
    elif realistic_r2 >= 0.65:
        print("   ⚠️  MODEL NEEDS MINOR FIXES")
    else:
        print("   ❌ MODEL NEEDS SIGNIFICANT WORK")

if __name__ == "__main__":
    main()
