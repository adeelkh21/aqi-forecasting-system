# Revised AQI Prediction Roadmap
## Focus: Simple, Effective, and Proven Approaches

### Current Status ✅
- [x] Data collection pipeline (150 days of data)
- [x] Basic feature engineering 
- [x] Enhanced feature engineering (needs simplification)
- [x] CI/CD pipeline for data collection
- [x] Data validation and quality checks

### Problem Analysis 🔍
**Current Issues:**
- Complex features causing NA/infinite values
- TFT model compatibility issues
- Poor R² performance with current approach
- Over-engineered features not helping

**Root Causes:**
- Too many complex derived features
- Model architectures too advanced for dataset size
- Need to focus on feature quality over quantity

---

## Phase 4 Revised: Simple & Effective Model Development

### Step 1: Feature Simplification & Selection 🎯
**Goal:** Keep only the most predictive, stable features

**Action Items:**
1. **Core Features Only:**
   - Direct measurements: `pm2_5`, `pm10`, `no2`, `o3`, `temperature`, `humidity`, `wind_speed`, `pressure`
   - Time features: `hour`, `day_of_week`, `is_weekend`
   - Simple lags: `pm2_5_24h_ago`, `pm10_24h_ago`, `aqi_numeric_24h_ago`
   - Rolling means: `pm2_5_6h_mean`, `pm10_6h_mean`, `wind_speed_6h_mean`

2. **Remove Complex Features:**
   - Physics-based features (causing infinities)
   - Fourier features (not helping with 150 days)
   - Complex statistical features (quantiles, skewness, kurtosis)
   - Interaction terms (for now)

3. **Feature Importance Analysis:**
   - Use simple correlation analysis
   - SHAP on baseline models only
   - Focus on top 15-20 features maximum

### Step 2: Proven Model Architectures 🚀
**Goal:** Use well-established, robust algorithms

**Model Selection:**
1. **Gradient Boosting Ensemble:**
   - **XGBoost** (proven for tabular data)
   - **LightGBM** (fast, efficient)
   - **CatBoost** (handles missing values well)

2. **Tree-Based Models:**
   - **Random Forest** (robust baseline)
   - **Extra Trees** (reduced overfitting)

3. **Linear Models (for comparison):**
   - **Ridge Regression** (with polynomial features)
   - **Elastic Net** (automatic feature selection)

**Why These Models:**
- Excellent for tabular time series data
- Handle missing values naturally
- Proven track record for air quality prediction
- Fast training and inference
- Good interpretability

### Step 3: Smart Ensemble Strategy 🎼
**Goal:** Combine models for better performance

**Ensemble Approaches:**
1. **Weighted Average Ensemble:**
   - Weight models by validation performance
   - Different weights for different horizons (24h, 48h, 72h)

2. **Stacking Ensemble:**
   - Use Ridge/Linear as meta-learner
   - Cross-validation for base model predictions

3. **Multi-Horizon Specialized Models:**
   - Separate models optimized for each prediction horizon
   - Short-term models (1-24h): Focus on recent lags
   - Long-term models (48-72h): Focus on trend features

### Step 4: Advanced Time Series Validation 📊
**Goal:** Robust evaluation that respects temporal order

**Validation Strategy:**
1. **Time Series Cross-Validation:**
   - Expanding window approach
   - Minimum 30 days training
   - 7-day validation windows
   - No data leakage

2. **Multi-Horizon Evaluation:**
   - Separate metrics for 24h, 48h, 72h predictions
   - R², MAE, RMSE for each horizon
   - Peak pollution prediction accuracy

3. **Real-World Testing:**
   - Walk-forward validation
   - Performance during high pollution events
   - Seasonal performance analysis

---

## Implementation Plan 📋

### Week 1: Feature Simplification
- [ ] Create `simple_features.py` with core features only
- [ ] Remove complex features from pipeline
- [ ] Validate data quality with simplified features
- [ ] Generate feature importance analysis

### Week 2: Model Development
- [ ] Implement optimized XGBoost/LightGBM models
- [ ] Create separate models for each prediction horizon
- [ ] Implement time series cross-validation
- [ ] Hyperparameter optimization with Optuna

### Week 3: Ensemble & Optimization
- [ ] Build weighted ensemble system
- [ ] Implement stacking approach
- [ ] Performance comparison and analysis
- [ ] Model interpretability analysis

### Week 4: Production & Monitoring
- [ ] Deploy best performing ensemble
- [ ] Real-time prediction API
- [ ] Performance monitoring dashboard
- [ ] Documentation and testing

---

## Expected Performance Targets 🎯

**Realistic Goals (based on literature):**
- **24-hour ahead:** R² ≥ 0.70, MAE ≤ 15
- **48-hour ahead:** R² ≥ 0.55, MAE ≤ 20  
- **72-hour ahead:** R² ≥ 0.40, MAE ≤ 25

**Success Criteria:**
- Consistent performance across seasons
- Good performance during pollution events
- Reliable uncertainty quantification
- Fast inference time (< 1 second)

---

## Key Success Factors 💡

1. **Simplicity First:** Start simple, add complexity only if it helps
2. **Data Quality:** Focus on clean, reliable features
3. **Domain Knowledge:** Use meteorological understanding
4. **Validation Rigor:** Never compromise on time series validation
5. **Interpretability:** Ensure models are explainable

This approach prioritizes **proven methods** over **cutting-edge complexity**, focusing on what actually works for AQI prediction with limited data.
