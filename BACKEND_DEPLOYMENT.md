# 🚀 Backend Deployment Guide

This guide will help you deploy your FastAPI backend so it's accessible from your Streamlit Cloud app.

## 🔍 **Current Issue:**

- ✅ **Frontend**: Streamlit app deployed on Streamlit Cloud
- ❌ **Backend**: FastAPI server offline (not accessible)

## 🚀 **Solution: Deploy Backend to Cloud Platform**

### **Option 1: Railway (Recommended - Free)**

Railway is a great free platform for hosting FastAPI backends.

#### **Step 1: Prepare for Railway**
1. **Files are ready**: `railway.json` and `Procfile` created
2. **CORS updated**: Backend now allows Streamlit Cloud domains
3. **Environment variables**: Configured for cloud deployment

#### **Step 2: Deploy to Railway**
1. **Go to** [Railway](https://railway.app/)
2. **Sign up** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**: `adeelkh21/aqi-forecasting-system`
6. **Set build command**: Leave empty (uses railway.json)
7. **Set start command**: Leave empty (uses railway.json)
8. **Click "Deploy"**

#### **Step 3: Get Your Backend URL**
1. **Wait for deployment** (2-3 minutes)
2. **Copy the generated URL** (e.g., `https://your-app.railway.app`)
3. **Update Streamlit app** with the new backend URL

### **Option 2: Render (Alternative - Free)**

Render is another excellent free platform.

#### **Step 1: Deploy to Render**
1. **Go to** [Render](https://render.com/)
2. **Sign up** with GitHub
3. **Click "New +" → "Web Service"**
4. **Connect your repository**
5. **Configure service**:
   - **Name**: `aqi-forecasting-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. **Click "Create Web Service"**

### **Option 3: Heroku (Alternative - Free Tier Discontinued)**

If you have existing Heroku credits.

## 🔧 **Update Streamlit App with Backend URL**

Once your backend is deployed, update your Streamlit app:

### **Method 1: Environment Variables in Streamlit Cloud**
1. **Go to** your Streamlit Cloud app
2. **Click "Settings"**
3. **Add secret**:
```toml
[backend]
url = "https://your-backend-url.railway.app"
```

### **Method 2: Update Code Directly**
Update `streamlit_app_clean.py`:
```python
# Replace localhost:8001 with your backend URL
BACKEND_URL = "https://your-backend-url.railway.app"

# Update all API calls
response = requests.get(f"{BACKEND_URL}/current-aqi", timeout=10)
```

## 📱 **Test Your Complete System**

### **1. Test Backend Health**
```bash
curl https://your-backend-url.railway.app/health
```

### **2. Test from Streamlit**
- Generate a forecast
- Check if backend responds
- Verify data collection works

## ⚠️ **Common Issues & Solutions**

### **Issue: CORS Errors**
**Solution**: Backend CORS is already configured for Streamlit Cloud

### **Issue: Backend Not Responding**
**Solution**: Check Railway/Render logs for errors

### **Issue: Models Not Loading**
**Solution**: First deployment takes 1-2 minutes for model training

### **Issue: Timeout Errors**
**Solution**: Increase timeout in Streamlit app to 120 seconds

## 🔄 **Automatic Updates**

### **Railway Auto-Deploy**
- Every push to GitHub automatically redeploys backend
- No manual intervention needed

### **Streamlit Auto-Deploy**
- Frontend updates automatically when you push changes

## 📊 **Monitoring Your Backend**

### **Railway Dashboard**
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory usage
- **Deployments**: Deployment history

### **Health Check Endpoint**
```bash
# Check if backend is running
curl https://your-backend-url.railway.app/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "backend": "FastAPI",
  "version": "1.0.0"
}
```

## 🎯 **Expected Results**

After deployment:
- ✅ **Backend**: Accessible via HTTPS URL
- ✅ **Frontend**: Can communicate with backend
- ✅ **Forecasting**: Works end-to-end
- ✅ **Data Collection**: Real-time data available
- ✅ **Complete System**: Fully functional

## 🚀 **Quick Deploy Commands**

### **Railway (Recommended)**
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repo
3. Deploy automatically

### **Manual Update**
```bash
# After backend deployment, update Streamlit
git add .
git commit -m "Update: Backend URL for production"
git push origin main
```

## 🎉 **Success!**

Once deployed:
- **Backend URL**: `https://your-app.railway.app`
- **Frontend URL**: `https://aqi-forecasting-system.streamlit.app`
- **Complete System**: Fully functional AQI forecasting

---

**Need help?** Check Railway/Render logs or contact their support!
