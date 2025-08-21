# GitHub Actions CI/CD Pipeline Setup

This repository includes automated CI/CD pipelines that run your forecasting system automatically.

## 🚀 Available Workflows

### 1. **Hourly Data Pipeline** (`.github/workflows/hourly_data_pipeline.yml`)
- **Frequency**: Every hour
- **Purpose**: Collect fresh data, preprocess, and select features
- **Best for**: Keeping data fresh and up-to-date

### 2. **Complete ML Pipeline** (`.github/workflows/complete_ml_pipeline.yml`)
- **Frequency**: Every 6 hours
- **Purpose**: Full pipeline including model fine-tuning and validation
- **Best for**: Complete system updates and model improvements

## ⚙️ Setup Instructions

### Step 1: Push Your Code to GitHub
```bash
# Add the new workflow files
git add .github/workflows/
git add GITHUB_ACTIONS_SETUP.md

# Commit and push
git commit -m "Add GitHub Actions CI/CD pipelines"
git push origin main
```

### Step 2: Configure Repository Secrets
Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

Add these secrets:
- **`OPENWEATHER_API_KEY`**: Your OpenWeatherMap API key
- **`METEOSTAT_API_KEY`**: Your Meteostat API key (if required)

### Step 3: Enable GitHub Actions
1. Go to **Actions** tab in your repository
2. You should see the workflows listed
3. Click on a workflow to see its runs

## 🔄 How It Works

### Hourly Data Pipeline:
1. **Data Collection**: Runs `01_data_collection.py` to fetch fresh data
2. **Preprocessing**: Runs `02_data_preprocessing.py` to create features
3. **Feature Selection**: Runs `03_feature_selection.py` to optimize features
4. **Auto-commit**: Commits and pushes updated data to repository

### Complete ML Pipeline:
1. **Data Pipeline**: Same as hourly + model fine-tuning
2. **Model Fine-tuning**: Runs `11_per_horizon_finetune.py`
3. **Daily Pipeline**: Runs `daily_runner.py` for model promotion
4. **Forecasting**: Generates new forecasts
5. **Validation**: Runs backtests to validate performance
6. **Auto-commit**: Commits data, models, and results

## 📊 Monitoring

### Check Pipeline Status:
- Go to **Actions** tab in your repository
- Click on a workflow to see detailed logs
- Green checkmark = success, red X = failure

### View Results:
- **Data**: Check `data_repositories/` folder
- **Models**: Check `saved_models/` folder
- **Forecasts**: Check `saved_models/forecasts/`
- **Reports**: Check `saved_models/reports/`

## 🚨 Troubleshooting

### Common Issues:

1. **API Key Errors**:
   - Verify secrets are set correctly
   - Check API key permissions and quotas

2. **Python Dependencies**:
   - Ensure `requirements.txt` is up-to-date
   - Check for version conflicts

3. **File Permission Errors**:
   - Workflows run on Linux (Ubuntu)
   - Some Windows-specific code may need adjustment

### Manual Trigger:
- Go to **Actions** → **Workflows**
- Click **Run workflow** button
- Select branch and click **Run workflow**

## 📈 Benefits

✅ **Automated Data Collection**: Fresh data every hour  
✅ **Continuous Model Improvement**: Models adapt to new patterns  
✅ **Performance Monitoring**: Regular backtests and validation  
✅ **Version Control**: All data and models tracked in Git  
✅ **Scalability**: Runs on GitHub's infrastructure  
✅ **Reliability**: Automatic retries and error handling  

## 🔧 Customization

### Change Frequency:
Edit the `cron` expression in workflow files:
- `'0 * * * *'` = Every hour
- `'0 */6 * * *'` = Every 6 hours
- `'0 0 * * *'` = Daily at midnight

### Add/Remove Steps:
Modify the workflow YAML files to include/exclude specific scripts or steps.

### Environment Variables:
Add more secrets and environment variables as needed for your specific setup.

## 🎯 Next Steps

1. **Push your code** to GitHub
2. **Set up API key secrets**
3. **Monitor the first pipeline run**
4. **Check results** in the repository
5. **Adjust frequency** if needed

Your forecasting system will now run automatically and continuously improve! 🚀
