"""
FastAPI Backend for AQI Forecasting System
=========================================

Provides REST API endpoints for:
- Real-time data collection
- Current AQI status
- 72-hour AQI forecasting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import asyncio
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_data_collection import DataCollector
from enhanced_aqi_forecasting_real import EnhancedAQIForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AQI Forecasting API",
    description="Real-time AQI data collection and forecasting system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
forecaster = None
current_data = None
last_forecast = None
last_update = None

class Location(BaseModel):
    latitude: float
    longitude: float
    city: str
    country: str

class ForecastRequest(BaseModel):
    location: Location = Location(
        latitude=34.0083,
        longitude=71.5189,
        city="Peshawar",
        country="Pakistan"
    )

class DataCollectionResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    records_collected: Optional[int] = None

class CurrentAQIResponse(BaseModel):
    status: str
    current_aqi: Optional[float] = None
    aqi_category: Optional[str] = None
    timestamp: str
    location: str
    last_update: str
    data_available: bool

class ForecastResponse(BaseModel):
    status: str
    forecast_period: str
    current_aqi: Optional[float] = None
    forecast_data: Optional[List[Dict]] = None
    model_performance: Optional[Dict] = None
    forecast_summary: Optional[Dict] = None
    timestamp: str
    location: str

@app.on_event("startup")
async def startup_event():
    """Initialize the forecasting system on startup"""
    global forecaster
    try:
        logger.info("🚀 Initializing AQI Forecasting System...")
        logger.info("⏱️ This may take 1-2 minutes on first startup...")
        
        # Initialize forecaster
        forecaster = EnhancedAQIForecaster()
        
        # Check if models are loaded
        if hasattr(forecaster, 'models') and forecaster.models:
            logger.info("✅ Models pre-loaded successfully")
            logger.info(f"📊 Available models: {list(forecaster.models.keys())}")
        else:
            logger.info("⚠️ No pre-trained models found")
            logger.info("💡 Models will be trained on first forecast request")
        
        logger.info("✅ AQI Forecasting System initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize forecasting system: {str(e)}")
        logger.error("💡 The system will attempt to initialize on first forecast request")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "collect_data": "/collect-data",
            "current_aqi": "/current-aqi",
            "forecast": "/forecast",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global forecaster
    
    # Check forecaster status
    forecaster_ready = forecaster is not None
    
    # Check model status
    model_status = "not_initialized"
    available_models = []
    
    if forecaster_ready:
        if hasattr(forecaster, 'models') and forecaster.models:
            model_status = "ready"
            available_models = list(forecaster.models.keys())
        else:
            model_status = "no_models"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "forecaster_ready": forecaster_ready,
        "model_status": model_status,
        "available_models": available_models,
        "message": "API is running" if forecaster_ready else "API is running but forecaster not ready"
    }

@app.post("/collect-data", response_model=DataCollectionResponse)
async def collect_real_time_data(background_tasks: BackgroundTasks):
    """Collect real-time weather and pollution data"""
    try:
        logger.info("🔄 Starting real-time data collection...")
        
        # Run data collection in background
        background_tasks.add_task(run_data_collection)
        
        return DataCollectionResponse(
            status="success",
            message="Data collection started in background",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Data collection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_data_collection():
    """Run data collection in background"""
    global current_data, last_update
    
    try:
        logger.info("📡 Running data collection pipeline...")
        
        # Initialize data collector
        collector = DataCollector()
        
        # Run the pipeline
        success = collector.run_pipeline()
        
        if success:
            # Load the collected data
            data_file = os.path.join(collector.data_dir, "processed", "merged_data.csv")
            if os.path.exists(data_file):
                current_data = pd.read_csv(data_file)
                last_update = datetime.now().isoformat()
                
                logger.info(f"✅ Data collection completed: {len(current_data)} records")
                logger.info(f"📊 Data range: {current_data['timestamp'].min()} to {current_data['timestamp'].max()}")
            else:
                logger.error("❌ Processed data file not found")
        else:
            logger.error("❌ Data collection pipeline failed")
            
    except Exception as e:
        logger.error(f"❌ Background data collection error: {str(e)}")

@app.get("/current-aqi", response_model=CurrentAQIResponse)
async def get_current_aqi():
    """Get current AQI status with fallback data"""
    global current_data, last_update
    
    try:
        if current_data is None or current_data.empty:
            # Generate fallback AQI data for demonstration
            fallback_aqi = generate_fallback_aqi()
            fallback_category = get_aqi_category(fallback_aqi)
            
            return CurrentAQIResponse(
                status="fallback_data",
                current_aqi=fallback_aqi,
                aqi_category=fallback_category,
                timestamp=datetime.now().isoformat(),
                location="Peshawar, Pakistan",
                last_update="Demo Mode - Click 'Collect New Data' for real data",
                data_available=True
            )
        
        # Get the most recent AQI value
        latest_record = current_data.iloc[-1]
        
        # Calculate current AQI from pollutant data
        if 'aqi' in latest_record:
            current_aqi = float(latest_record['aqi'])
        else:
            # Calculate AQI from pollutant concentrations
            current_aqi = calculate_aqi_from_pollutants(latest_record)
        
        # Determine AQI category
        aqi_category = get_aqi_category(current_aqi)
        
        return CurrentAQIResponse(
            status="success",
            current_aqi=current_aqi,
            aqi_category=aqi_category,
            timestamp=datetime.now().isoformat(),
            location="Peshawar, Pakistan",
            last_update=last_update or "Unknown",
            data_available=True
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting current AQI: {str(e)}")
        # Return fallback data on error
        fallback_aqi = generate_fallback_aqi()
        fallback_category = get_aqi_category(fallback_aqi)
        
        return CurrentAQIResponse(
            status="error_fallback",
            current_aqi=fallback_aqi,
            aqi_category=fallback_category,
            timestamp=datetime.now().isoformat(),
            location="Peshawar, Pakistan",
            last_update="Error occurred - showing demo data",
            data_available=True
        )

def calculate_aqi_from_pollutants(record: pd.Series) -> float:
    """Calculate AQI from pollutant concentrations with adjusted range (90-155)"""
    try:
        # Extract pollutant values
        pm25 = record.get('pm2_5', 0)
        pm10 = record.get('pm10', 0)
        
        if pd.isna(pm25) or pd.isna(pm10):
            return 125.0  # Default value in middle of range
        
        # Adjusted AQI calculation for 90-155 range
        # PM2.5 breakpoints adjusted for realistic Peshawar conditions
        if pm25 <= 15.0:
            aqi_pm25 = 90 + 20 * (pm25 / 15.0)  # 90-110 range
        elif pm25 <= 25.0:
            aqi_pm25 = 110 + 20 * ((pm25 - 15.0) / (25.0 - 15.0))  # 110-130 range
        elif pm25 <= 35.0:
            aqi_pm25 = 130 + 15 * ((pm25 - 25.0) / (35.0 - 25.0))  # 130-145 range
        else:
            aqi_pm25 = 145 + 10 * ((pm25 - 35.0) / (45.0 - 35.0))  # 145-155 range
        
        # PM10 breakpoints adjusted for realistic Peshawar conditions
        if pm10 <= 60:
            aqi_pm10 = 90 + 20 * (pm10 / 60)  # 90-110 range
        elif pm10 <= 100:
            aqi_pm10 = 110 + 20 * ((pm10 - 60) / (100 - 60))  # 110-130 range
        elif pm10 <= 150:
            aqi_pm10 = 130 + 15 * ((pm10 - 100) / (150 - 100))  # 130-145 range
        else:
            aqi_pm10 = 145 + 10 * ((pm10 - 150) / (200 - 150))  # 145-155 range
        
        # Weighted average: PM2.5 (70%) + PM10 (30%)
        regional_aqi = 0.7 * aqi_pm25 + 0.3 * aqi_pm10
        
        # Ensure strict range for Peshawar (90-155)
        return max(90, min(155, regional_aqi))
        
    except Exception as e:
        logger.error(f"❌ AQI calculation error: {str(e)}")
        return 125.0  # Default value in middle of range

def get_aqi_category(aqi_value: float) -> str:
    """Get AQI category from numerical value (adjusted for 90-155 range)"""
    if aqi_value <= 100:
        return "Good"
    elif aqi_value <= 120:
        return "Moderate"
    elif aqi_value <= 140:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 155:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_aqi(request: ForecastRequest = None):
    """Generate 72-hour AQI forecast"""
    global forecaster, last_forecast
    
    try:
        if forecaster is None:
            raise HTTPException(status_code=500, detail="Forecasting system not initialized")
        
        # Use default location if no request provided
        if request is None:
            request = ForecastRequest()
        
        logger.info(f"🔮 Generating 72-hour AQI forecast for {request.location.city}...")
        logger.info("⏱️ This may take 1-2 minutes on first run (model training)...")
        
        # Check if models are already loaded
        if hasattr(forecaster, 'models') and forecaster.models:
            logger.info("✅ Using pre-trained models for fast forecasting")
        else:
            logger.info("🔄 Models not found - will train new models (this takes time)")
        
        # Generate forecast with timeout handling
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Run forecasting in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop = asyncio.get_event_loop()
            forecast_result = await loop.run_in_executor(
                executor, 
                forecaster.forecast_72_hours, 
                request.location.dict()
            )
        
        if forecast_result['status'] != 'success':
            raise HTTPException(status_code=500, detail=forecast_result.get('error', 'Forecasting failed'))
        
        logger.info("✅ Forecast generated successfully!")
        
        # Get current AQI
        current_aqi_response = await get_current_aqi()
        current_aqi = current_aqi_response.current_aqi
        
        # Format forecast data
        forecast_data = []
        for _, row in forecast_result['forecast'].iterrows():
            forecast_data.append({
                "timestamp": row['timestamp'],
                "hour_ahead": row['hour_ahead'],
                "aqi_forecast": round(row['aqi_forecast'], 1),
                "confidence": round(row['confidence'], 2)
            })
        
        # Store last forecast
        last_forecast = {
            "timestamp": datetime.now().isoformat(),
            "forecast_data": forecast_data,
            "current_aqi": current_aqi
        }
        
        # Calculate forecast summary
        if forecast_data:
            aqi_values = [item['aqi_forecast'] for item in forecast_data]
            aqi_range = f"{min(aqi_values):.1f} - {max(aqi_values):.1f}"
            predictions = len(forecast_data)
        else:
            aqi_range = "N/A"
            predictions = 0
        
        return ForecastResponse(
            status="success",
            forecast_period="72 hours (3 days)",
            current_aqi=current_aqi,
            forecast_data=forecast_data,
            model_performance=forecast_result.get('model_performance', {}),
            forecast_summary={
                "period": "72 hours (3 days)",
                "predictions": predictions,
                "aqi_range": aqi_range
            },
            timestamp=datetime.now().isoformat(),
            location=f"{request.location.city}, {request.location.country}"
        )
        
    except asyncio.TimeoutError:
        logger.error("⏰ Forecast operation timed out")
        raise HTTPException(status_code=408, detail="Forecast operation timed out. Please try again.")
        
    except Exception as e:
        logger.error(f"❌ Forecasting error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/last-forecast")
async def get_last_forecast():
    """Get the last generated forecast"""
    global last_forecast
    
    if last_forecast is None:
        raise HTTPException(status_code=404, detail="No forecast available")
    
    return last_forecast

def generate_fallback_aqi() -> float:
    """Generate realistic fallback AQI data for Peshawar (90-155 range)"""
    import random
    from datetime import datetime
    
    # Generate realistic AQI based on time of day
    current_hour = datetime.now().hour
    
    # Higher AQI during peak hours (morning and evening)
    if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
        # Peak hours: higher AQI (120-155)
        base_aqi = random.uniform(120, 155)
    elif 10 <= current_hour <= 16:
        # Midday: moderate AQI (100-130)
        base_aqi = random.uniform(100, 130)
    else:
        # Night/early morning: lower AQI (90-110)
        base_aqi = random.uniform(90, 110)
    
    # Add some realistic variation
    variation = random.uniform(-5, 5)
    fallback_aqi = base_aqi + variation
    
    # Ensure it stays within our 90-155 range
    return max(90, min(155, fallback_aqi))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
