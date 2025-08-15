# 🚀 Deployment Guide

This guide will help you deploy the AQI Forecasting System to GitHub and Streamlit Cloud.

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free)
- Git installed on your local machine
- Python 3.8+ installed

## 🔧 Step 1: Prepare Your Local Repository

### 1.1 Initialize Git Repository
```bash
# Navigate to your project directory
cd FinalIA

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: AQI Forecasting System"
```

### 1.2 Create GitHub Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name: `aqi-forecasting-system`
5. Description: `Real-Time AQI Forecasting System with ML`
6. Make it **Public** (required for Streamlit Cloud free tier)
7. Don't initialize with README (we already have one)
8. Click "Create repository"

### 1.3 Connect and Push to GitHub
```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/aqi-forecasting-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🌐 Step 2: Deploy to Streamlit Cloud

### 2.1 Connect Streamlit Cloud to GitHub
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/aqi-forecasting-system`
5. Set the main file path: `streamlit_app_clean.py`
6. Click "Deploy!"

### 2.2 Configure Streamlit Cloud Settings
In your Streamlit Cloud app settings:

**Advanced Settings:**
- Python version: 3.9
- Streamlit version: 1.27.0
- Add packages: (leave empty, will use requirements.txt)

**Secrets:**
```toml
# Add any API keys or sensitive data here
[api_keys]
openweather_api_key = "your_api_key_here"
meteostat_api_key = "your_api_key_here"
```

## 🔄 Step 3: Continuous Deployment

### 3.1 Automatic Deployment
- Every time you push to the `main` branch, Streamlit Cloud will automatically redeploy
- You can also manually trigger deployments from the Streamlit Cloud dashboard

### 3.2 Deployment Commands
```bash
# Make changes to your code
# Then commit and push
git add .
git commit -m "Update: Enhanced weather display and analytics"
git push origin main

# Streamlit Cloud will automatically redeploy
```

## 📁 Step 4: Repository Structure for Deployment

Ensure your repository has this structure:
```
aqi-forecasting-system/
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── api/
│   └── main.py             # FastAPI backend
├── data_repositories/       # Data storage (excluded from git)
├── saved_models/           # Trained models (excluded from git)
├── streamlit_app_clean.py  # Main Streamlit app
├── enhanced_aqi_forecasting_real.py
├── phase1_data_collection.py
├── train_and_save_models.py
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
└── DEPLOYMENT.md           # This file
```

## ⚠️ Important Notes for Deployment

### 4.1 Data Files
- **Don't commit large data files** (CSV, models, etc.)
- Use `.gitignore` to exclude them
- Streamlit Cloud will handle dependencies from `requirements.txt`

### 4.2 Model Files
- **Don't commit trained models** (they're large and change frequently)
- Models will be trained on first run in Streamlit Cloud
- This may take 1-2 minutes on first deployment

### 4.3 API Keys
- Store sensitive data in Streamlit Cloud secrets
- Never commit API keys to GitHub
- Use environment variables in production

## 🚀 Step 5: Post-Deployment

### 5.1 Test Your App
1. Visit your Streamlit Cloud URL
2. Test all features:
   - Dashboard loading
   - Weather display
   - Forecast generation
   - Analytics page
   - Historical EDA

### 5.2 Monitor Performance
- Check Streamlit Cloud logs for errors
- Monitor app performance and response times
- Ensure all dependencies are loading correctly

### 5.3 Troubleshooting
Common issues and solutions:

**Issue: Models not loading**
- Solution: Wait 1-2 minutes for first-time model training
- Check Streamlit Cloud logs for errors

**Issue: API connection errors**
- Solution: Ensure FastAPI backend is accessible
- Check CORS settings if needed

**Issue: Missing dependencies**
- Solution: Verify `requirements.txt` includes all packages
- Check Streamlit Cloud package installation logs

## 🔧 Step 6: Production Considerations

### 6.1 Environment Variables
Set these in Streamlit Cloud secrets:
```toml
[production]
debug_mode = false
log_level = "info"
api_timeout = 120
```

### 6.2 Performance Optimization
- Enable caching in Streamlit
- Optimize data loading
- Use efficient chart rendering

### 6.3 Security
- Disable debug mode in production
- Use HTTPS (automatic with Streamlit Cloud)
- Implement rate limiting if needed

## 📊 Step 7: Monitoring and Maintenance

### 7.1 Regular Updates
```bash
# Update your local repository
git pull origin main

# Make changes and push
git add .
git commit -m "Update: Performance improvements"
git push origin main
```

### 7.2 Version Management
- Use semantic versioning for releases
- Tag important releases
- Keep a changelog

### 7.3 Backup Strategy
- Regular commits to GitHub
- Backup important configuration files
- Document deployment procedures

## 🎉 Congratulations!

Your AQI Forecasting System is now deployed and accessible worldwide through Streamlit Cloud!

### **Your App URL:**
```
https://your-app-name.streamlit.app
```

### **Next Steps:**
1. Share your app with others
2. Monitor performance and usage
3. Gather feedback and improve
4. Consider adding more features
5. Scale up if needed

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Review GitHub Issues
3. Check the troubleshooting section
4. Contact Streamlit support if needed

---

**Happy Deploying! 🚀**
