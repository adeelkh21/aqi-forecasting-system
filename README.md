# 🌤️ Real-Time AQI Forecasting System

A comprehensive air quality forecasting system built with Python, FastAPI, and Streamlit that provides real-time AQI predictions for Peshawar, Pakistan using advanced machine learning models.

## 🚀 Features

### **Core Functionality**
- **Real-time AQI Forecasting**: 72-hour predictions using ensemble ML models
- **Live Weather Integration**: Current weather conditions and hourly forecasts
- **Advanced ML Models**: LSTM, Random Forest, Gradient Boosting, Prophet, SARIMA
- **Interactive Dashboard**: Beautiful, responsive Streamlit interface
- **RESTful API**: FastAPI backend with comprehensive endpoints

### **Machine Learning Capabilities**
- **Ensemble Forecasting**: Combines multiple models for optimal predictions
- **Feature Engineering**: Advanced temporal and environmental features
- **Model Persistence**: Trained models saved and loaded automatically
- **Performance Metrics**: R² scores, MAE, RMSE, MAPE analysis
- **Real-time Training**: Models can be retrained with new data

### **Data Sources**
- **Historical Data**: Real air quality data from Peshawar
- **Weather Data**: Meteostat integration for meteorological data
- **Pollution Data**: PM2.5, PM10, NO2, O3, SO2, CO monitoring
- **Real-time Collection**: Automated data collection pipeline

## 🏗️ Architecture

```
FinalIA/
├── api/                    # FastAPI backend
│   └── main.py           # Main API endpoints
├── data_repositories/     # Data storage
│   └── historical_data/   # Historical datasets
├── saved_models/          # Trained ML models
├── streamlit_app_clean.py # Main Streamlit dashboard
├── enhanced_aqi_forecasting_real.py  # Core ML engine
├── phase1_data_collection.py        # Data collection
├── train_and_save_models.py         # Model training
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- pip package manager
- Git

### **Setup Instructions**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/aqi-forecasting-system.git
cd aqi-forecasting-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the system**
```bash
# Terminal 1: Start FastAPI backend
cd api
uvicorn main:app --host 0.0.0.0 --port 8001

# Terminal 2: Start Streamlit frontend
streamlit run streamlit_app_clean.py
```

## 📊 Usage

### **Dashboard Features**
- **Real-time AQI Monitoring**: Live air quality status
- **Weather Display**: Current conditions and hourly forecasts
- **Forecast Generation**: 72-hour AQI predictions
- **Historical Analysis**: Comprehensive EDA and insights
- **Model Performance**: Real-time ML model metrics

### **API Endpoints**
- `GET /`: API information
- `GET /health`: System health check
- `POST /collect-data`: Trigger data collection
- `GET /current-aqi`: Get current AQI status
- `POST /forecast`: Generate 72-hour forecast
- `GET /last-forecast`: Retrieve last forecast

### **Model Training**
```python
# Train and save models
python train_and_save_models.py

# Models will be saved to saved_models/ directory
```

## 🔧 Configuration

### **Environment Variables**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8001

# Data Collection
COLLECTION_DAYS=7
LOCATION_LAT=34.0083
LOCATION_LON=71.5189
```

### **Model Parameters**
- **LSTM**: 50 units, 2 layers, dropout 0.2
- **Random Forest**: 100 estimators, max_depth 15
- **Gradient Boosting**: 100 estimators, learning_rate 0.1
- **Prophet**: Yearly seasonality, weekly seasonality
- **SARIMA**: (1,1,1)(1,1,1,24) configuration

## 📈 Performance

### **Model Accuracy (R² Scores)**
- **Random Forest**: 0.85+
- **Gradient Boosting**: 0.83+
- **LSTM**: 0.80+
- **Prophet**: 0.75+
- **SARIMA**: 0.70+

### **Forecast Horizon**
- **Short-term**: 1-24 hours (high accuracy)
- **Medium-term**: 24-48 hours (good accuracy)
- **Long-term**: 48-72 hours (moderate accuracy)

## 🚀 Deployment

### **Streamlit Cloud Deployment**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set deployment configuration
4. Deploy automatically

### **Local Deployment**
```bash
# Production mode
streamlit run streamlit_app_clean.py --server.port 8501 --server.address 0.0.0.0

# With custom theme
streamlit run streamlit_app_clean.py --theme.base light
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app_clean.py", "--server.port=8501"]
```

## 🔍 Troubleshooting

### **Common Issues**
1. **Models not loading**: Run `train_and_save_models.py` first
2. **API connection error**: Ensure FastAPI backend is running on port 8001
3. **Data collection failed**: Check internet connection and API keys
4. **Forecast timeout**: First run takes 1-2 minutes for model training

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Documentation

### **Forecast Request**
```json
{
  "location": {
    "latitude": 34.0083,
    "longitude": 71.5189,
    "city": "Peshawar",
    "country": "Pakistan"
  }
}
```

### **Forecast Response**
```json
{
  "status": "success",
  "forecast_period": "72 hours (3 days)",
  "current_aqi": 125.5,
  "forecast_data": [...],
  "model_performance": {...},
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Meteostat, OpenWeatherMap
- **ML Libraries**: TensorFlow, Scikit-learn, Prophet, Statsmodels
- **Web Framework**: FastAPI, Streamlit
- **Research**: EPA AQI standards, regional air quality studies

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/aqi-forecasting-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/aqi-forecasting-system/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] Mobile app development
- [ ] Additional ML models (XGBoost, LightGBM)
- [ ] Real-time notifications
- [ ] Multi-city support
- [ ] Advanced visualization options
- [ ] API rate limiting and authentication
- [ ] Database integration (PostgreSQL, MongoDB)

---

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ for better air quality monitoring**
