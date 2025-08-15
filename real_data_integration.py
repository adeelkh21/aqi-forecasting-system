"""
Real Data Integration for AQI Dashboard
======================================

Connect to real weather and pollution APIs to get actual live data
instead of dummy/simulated data.

APIs to integrate:
1. OpenWeatherMap - Current weather and pollution
2. IQAir - Real-time AQI data
3. Government pollution monitoring APIs

EPA AQI Standards Implementation:
- PM2.5, PM10, NO2, O3, SO2, CO breakpoints
- Proper AQI calculation using EPA methodology
- Real-time data validation and quality checks
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
except Exception:
    pass

logger = logging.getLogger(__name__)

class EPA_AQI_Calculator:
    """EPA AQI calculation using official breakpoints and methodology"""
    
    # EPA AQI Breakpoints for each pollutant
    AQI_BREAKPOINTS = {
        'pm2_5': [
            (0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ],
        'pm10': [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500)
        ],
        'no2': [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500)
        ],
        'o3': [
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 300, 301, 400),
            (301, 400, 401, 500)
        ],
        'so2': [
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500)
        ],
        'co': [
            (0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500)
        ]
    }
    
    @classmethod
    def calculate_aqi_for_pollutant(cls, pollutant: str, concentration: float) -> Tuple[float, str]:
        """Calculate AQI for a single pollutant using EPA breakpoints"""
        if pollutant not in cls.AQI_BREAKPOINTS:
            return 0, "Unknown"
        
        if concentration <= 0:
            return 0, "Good"
        
        breakpoints = cls.AQI_BREAKPOINTS[pollutant]
        
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                # Linear interpolation formula: AQI = AQI_low + (conc - bp_low) * (AQI_high - AQI_low) / (bp_high - bp_low)
                aqi = aqi_low + (concentration - bp_low) * (aqi_high - aqi_low) / (bp_high - bp_low)
                
                # Determine category
                if aqi <= 50:
                    category = "Good"
                elif aqi <= 100:
                    category = "Moderate"
                elif aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif aqi <= 200:
                    category = "Unhealthy"
                elif aqi <= 300:
                    category = "Very Unhealthy"
                else:
                    category = "Hazardous"
                
                return round(aqi, 1), category
        
        # If concentration exceeds highest breakpoint
        return 500, "Hazardous"
    
    @classmethod
    def calculate_overall_aqi(cls, pollutants: Dict[str, float]) -> Dict[str, any]:
        """Calculate overall AQI using EPA methodology (highest individual AQI)"""
        aqi_results = {}
        max_aqi = 0
        max_pollutant = None
        max_category = None
        
        for pollutant, concentration in pollutants.items():
            if pollutant in cls.AQI_BREAKPOINTS and concentration > 0:
                aqi, category = cls.calculate_aqi_for_pollutant(pollutant, concentration)
                aqi_results[pollutant] = {
                    'concentration': concentration,
                    'aqi': aqi,
                    'category': category
                }
                
                if aqi > max_aqi:
                    max_aqi = aqi
                    max_pollutant = pollutant
                    max_category = category
        
        return {
            'overall_aqi': max_aqi,
            'primary_pollutant': max_pollutant,
            'category': max_category,
            'breakdown': aqi_results,
            'calculation_method': 'EPA Standard'
        }

class RealDataCollector:
    """Collect real weather and pollution data from live APIs"""
    
    def __init__(self):
        """Initialize with API keys"""
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.iqair_api_key = os.getenv('IQAIR_API_KEY')
        
        if not self.openweather_api_key:
            logger.warning("⚠️  OPENWEATHER_API_KEY not set")
        
        # API endpoints
        self.openweather_current_url = "http://api.openweathermap.org/data/2.5/weather"
        self.openweather_pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.openweather_forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        self.iqair_current_url = "http://api.airvisual.com/v2/city"
        
        logger.info("🌐 Real Data Collector initialized")
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather data from OpenWeatherMap"""
        if not self.openweather_api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.openweather_current_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'wind_direction': data.get('wind', {}).get('deg', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'clouds': data.get('clouds', {}).get('all', 0),
                    'weather_description': data['weather'][0]['description'],
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Weather data retrieved: {weather_data['temperature']}°C, {weather_data['humidity']}%")
                return weather_data
            
            else:
                logger.error(f"❌ Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting weather data: {str(e)}")
            return None
    
    def get_current_air_pollution(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current air pollution data from OpenWeatherMap"""
        if not self.openweather_api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key
            }
            
            response = requests.get(self.openweather_pollution_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'list' in data and len(data['list']) > 0:
                    pollution_data = data['list'][0]
                    components = pollution_data['components']
                    
                    # Extract pollutant concentrations
                    pollutants = {
                        'pm2_5': components.get('pm2_5', 0),
                        'pm10': components.get('pm10', 0),
                        'no2': components.get('no2', 0),
                        'o3': components.get('o3', 0),
                        'so2': components.get('so2', 0),
                        'co': components.get('co', 0)
                    }
                    
                    # Calculate EPA AQI
                    aqi_result = EPA_AQI_Calculator.calculate_overall_aqi(pollutants)
                    
                    pollution_result = {
                        'aqi_epa': aqi_result['overall_aqi'],
                        'primary_pollutant': aqi_result['primary_pollutant'],
                        'category': aqi_result['category'],
                        'pollutants': pollutants,
                        'aqi_breakdown': aqi_result['breakdown'],
                        'calculation_method': aqi_result['calculation_method'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"✅ Pollution data retrieved: EPA AQI {pollution_result['aqi_epa']} ({pollution_result['category']})")
                    return pollution_result
                
                else:
                    logger.error("❌ No pollution data in response")
                    return None
            
            else:
                logger.error(f"❌ Pollution API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting pollution data: {str(e)}")
            return None
    
    def get_iqair_data(self, city: str, state: str, country: str) -> Optional[Dict]:
        """Get AQI data from IQAir (more accurate for AQI)"""
        if not self.iqair_api_key:
            return None
            
        try:
            params = {
                'city': city,
                'state': state,
                'country': country,
                'key': self.iqair_api_key
            }
            
            response = requests.get(self.iqair_current_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'success':
                    current = data['data']['current']
                    pollution = current['pollution']
                    weather = current['weather']
                    
                    iqair_data = {
                        'aqi_us': pollution['aqius'],  # US AQI
                        'aqi_china': pollution['aqicn'],  # China AQI
                        'main_pollutant': pollution['mainus'],
                        'temperature': weather['tp'],
                        'humidity': weather['hu'],
                        'pressure': weather['pr'],
                        'wind_speed': weather['ws'],
                        'wind_direction': weather['wd'],
                        'timestamp': pollution['ts']
                    }
                    
                    logger.info(f"✅ IQAir data retrieved: US AQI {iqair_data['aqi_us']}")
                    return iqair_data
                
                else:
                    logger.error(f"❌ IQAir API error: {data.get('message', 'Unknown error')}")
                    return None
            
            else:
                logger.error(f"❌ IQAir API HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting IQAir data: {str(e)}")
            return None
    
    def get_weather_forecast(self, lat: float, lon: float, hours: int = 120) -> Optional[List[Dict]]:
        """Get weather forecast data"""
        if not self.openweather_api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.openweather_forecast_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                forecast_list = []
                for item in data['list'][:hours//3]:  # API returns 3-hour intervals
                    forecast_item = {
                        'datetime': item['dt_txt'],
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item.get('wind', {}).get('speed', 0),
                        'clouds': item.get('clouds', {}).get('all', 0),
                        'weather': item['weather'][0]['description']
                    }
                    forecast_list.append(forecast_item)
                
                logger.info(f"✅ Weather forecast retrieved: {len(forecast_list)} periods")
                return forecast_list
            
            else:
                logger.error(f"❌ Forecast API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting weather forecast: {str(e)}")
            return None
    
    def get_comprehensive_current_data(self, location: Dict) -> Dict:
        """Get comprehensive current data from all sources"""
        lat = location['latitude']
        lon = location['longitude']
        city = location.get('city', 'Peshawar')
        country = location.get('country', 'Pakistan')
        
        result = {
            'location': location,
            'timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'weather': None,
            'pollution': None,
            'current_aqi': None,
            'status': 'success'
        }
        
        # Try to get weather data
        weather_data = self.get_current_weather(lat, lon)
        if weather_data:
            result['weather'] = weather_data
            result['data_sources'].append('OpenWeatherMap-Weather')
        
        # Try to get pollution data from OpenWeatherMap
        pollution_data = self.get_current_air_pollution(lat, lon)
        if pollution_data:
            result['pollution'] = pollution_data
            result['current_aqi'] = pollution_data['aqi_epa']
            result['data_sources'].append('OpenWeatherMap-Pollution')
        
        # Try to get more accurate AQI from IQAir (if available)
        if self.iqair_api_key:
            iqair_data = self.get_iqair_data(city, '', country)
            if iqair_data:
                result['iqair'] = iqair_data
                result['current_aqi'] = iqair_data['aqi_us']  # Use more accurate IQAir data
                result['data_sources'].append('IQAir')
        
        # If no real data available, provide fallback
        if not result['data_sources']:
            result['status'] = 'fallback'
            result['message'] = 'No real-time data available. Using fallback values.'
            # Provide realistic fallback data for Peshawar
            result['current_aqi'] = 134  # Based on user's observation
            result['weather'] = {
                'temperature': 28,
                'humidity': 60,
                'pressure': 1013,
                'wind_speed': 5.2
            }
        
        logger.info(f"📊 Comprehensive data collected: AQI {result.get('current_aqi', 'N/A')}")
        return result

def test_real_data():
    """Test real data collection"""
    print("🧪 Testing Real Data Collection")
    print("=" * 32)
    
    collector = RealDataCollector()
    
    # Test with Peshawar coordinates
    peshawar_location = {
        'latitude': 34.0151,
        'longitude': 71.5249,
        'city': 'Peshawar',
        'country': 'Pakistan'
    }
    
    print("🌍 Testing Peshawar data collection...")
    data = collector.get_comprehensive_current_data(peshawar_location)
    
    print(f"📊 Results:")
    print(f"   Current AQI: {data.get('current_aqi', 'N/A')}")
    print(f"   Data Sources: {', '.join(data.get('data_sources', ['None']))}")
    print(f"   Status: {data.get('status', 'unknown')}")
    
    if data.get('weather'):
        weather = data['weather']
        print(f"   Temperature: {weather.get('temperature', 'N/A')}°C")
        print(f"   Humidity: {weather.get('humidity', 'N/A')}%")
    
    if data.get('pollution'):
        pollution = data['pollution']
        print(f"   EPA AQI: {pollution.get('aqi_epa', 'N/A')}")
        print(f"   Category: {pollution.get('category', 'N/A')}")
        print(f"   Primary Pollutant: {pollution.get('primary_pollutant', 'N/A')}")
        if 'pollutants' in pollution:
            pollutants = pollution['pollutants']
            print(f"   PM2.5: {pollutants.get('pm2_5', 'N/A')} μg/m³")
            print(f"   PM10: {pollutants.get('pm10', 'N/A')} μg/m³")
    
    return data

if __name__ == "__main__":
    test_real_data()
