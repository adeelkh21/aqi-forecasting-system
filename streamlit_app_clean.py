"""
AQI Forecasting Dashboard - Real-Time Interactive Version
======================================================

A truly interactive and real-time dashboard that automatically:
- Collects data every few minutes
- Applies forecasting continuously
- Shows current AQI based on actual time
- Updates in real-time without manual intervention
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import numpy as np
import asyncio
import threading
from streamlit_autorefresh import st_autorefresh
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time AQI Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8001"

# Auto-refresh every 2 minutes (120 seconds)
st_autorefresh(interval=120000, key="data_refresh")

# Header with author information and hyperlinks
def show_header():
    """Display header with author information and hyperlinks"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; text-align: center; color: white; 
                margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h1 style="margin: 0 0 1rem 0; font-size: 2.5rem; font-weight: bold; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌤️ Real-Time AQI Forecasting System
        </h1>
        <p style="font-size: 1.2rem; margin-bottom: 1.5rem; opacity: 0.9;">
            Advanced Air Quality Index Forecasting for Peshawar, Pakistan
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">👨‍💻 Created by</h3>
                <p style="margin: 0; font-size: 1.3rem; font-weight: bold;">Muhammad Adeel</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🔗 Connect</h3>
                <p style="margin: 0;">
                    <a href="https://www.linkedin.com/in/muhammadadeel21" 
                       target="_blank" 
                       style="color: #00d4ff; text-decoration: none; font-weight: bold;">
                        LinkedIn: muhammadadeel21
                    </a>
                </p>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🐙 GitHub</h3>
                <p style="margin: 0;">
                    <a href="https://github.com/adeelkh21" 
                       target="_blank" 
                       style="color: #00d4ff; text-decoration: none; font-weight: bold;">
                        adeelkh21
                    </a>
                </p>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <p style="margin: 0; font-size: 1rem; opacity: 0.8;">
                🤖 Machine Learning • 📊 Real-Time Data • 🌍 Environmental Monitoring
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.live-indicator {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 25px;
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.warning-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.metric-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.info-box {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #333;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.weather-card {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.weather-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

.status-good { color: #00b894; font-weight: bold; }
.status-moderate { color: #fdcb6e; font-weight: bold; }
.status-unhealthy-for-sensitive-groups { color: #e17055; font-weight: bold; }
.status-unhealthy { color: #d63031; font-weight: bold; }
.status-very-unhealthy { color: #6c5ce7; font-weight: bold; }
.status-hazardous { color: #2d3436; font-weight: bold; }
.status-warning { color: #f39c12; font-weight: bold; }

.pulse { animation: pulse 2s infinite; }
.blink { animation: blink 1.5s infinite; }
.glow { animation: glow 2s ease-in-out infinite alternate; }
.shake { animation: shake 0.5s ease-in-out infinite; }

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
}

@keyframes glow {
    from { box-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #e60073; }
    to { box-shadow: 0 0 10px #fff, 0 0 20px #ff4da6, 0 0 30px #ff4da6; }
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}
</style>
""", unsafe_allow_html=True)

# Global state management
if 'last_data_collection' not in st.session_state:
    st.session_state.last_data_collection = None
if 'last_forecast' not in st.session_state:
    st.session_state.last_forecast = None
if 'current_aqi_data' not in st.session_state:
    st.session_state.current_aqi_data = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = True

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def collect_real_time_data():
    """Collect real-time data from the API"""
    try:
        response = requests.post(f"{API_BASE_URL}/collect-data", timeout=30)
        if response.status_code == 200:
            st.session_state.last_data_collection = datetime.now()
            return response.json()
        else:
            st.error(f"Data collection failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error collecting data: {str(e)}")
        return None

def get_current_aqi():
    """Get current AQI from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/current-aqi", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get current AQI: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting current AQI: {str(e)}")
        return None

def get_forecast():
    """Get 72-hour AQI forecast from the API"""
    try:
        # Show loading indicator
        with st.spinner("🔮 Generating 72-hour AQI forecast... This may take 1-2 minutes on first run."):
            # Prepare request data
            request_data = {
                "location": {
                    "latitude": 34.0083,
                    "longitude": 71.5189,
                    "city": "Peshawar",
                    "country": "Pakistan"
                }
            }
            
            # Make request with increased timeout
            response = requests.post(
                "http://localhost:8001/forecast",
                json=request_data,
                timeout=120  # Increased timeout to 2 minutes
            )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.forecast_data = result
            st.session_state.last_forecast = datetime.now()
            st.success("✅ Forecast generated successfully!")
            return result
        else:
            st.error(f"❌ Forecast failed with status {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ **Forecast Request Timed Out**")
        st.warning("The forecasting process is taking longer than expected. This usually happens when:")
        st.write("• Models are being retrained for the first time")
        st.write("• Large datasets are being processed")
        st.write("• System resources are limited")
        st.info("**Solutions:**")
        st.write("1. Wait a few minutes and try again")
        st.write("2. Check if the backend API is running smoothly")
        st.write("3. Ensure models are already trained and saved")
        st.write("4. **First-time setup**: Models need to be trained once, subsequent forecasts will be faster")
        return None
        
    except requests.exceptions.ConnectionError:
        st.error("🔌 **Connection Error**")
        st.warning("Cannot connect to the forecasting API. Please ensure:")
        st.write("• The backend API is running on port 8001")
        st.write("• No firewall is blocking the connection")
        st.write("• The API service is healthy")
        return None
        
    except Exception as e:
        st.error(f"❌ **Error getting forecast**: {str(e)}")
        st.info("Please check the console for detailed error information.")
        return None

def get_aqi_category(aqi_value):
    """Get AQI category based on value (adjusted for 90-155 range)"""
    if aqi_value is None:
        return "Unknown"
    try:
        aqi_value = float(aqi_value)
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
    except (ValueError, TypeError):
        return "Invalid"

def get_aqi_color(aqi_value):
    """Get color for AQI value (adjusted for 90-155 range)"""
    if aqi_value is None:
        return "#808080"
    try:
        aqi_value = float(aqi_value)
        if aqi_value <= 100:
            return "#00E400"  # Green
        elif aqi_value <= 120:
            return "#FFA500"  # Orange
        elif aqi_value <= 140:
            return "#FF7E00"  # Dark Orange
        elif aqi_value <= 155:
            return "#FF0000"  # Red
        else:
            return "#8B0000"  # Dark Red
    except (ValueError, TypeError):
        return "#808080"

def create_aqi_gauge(aqi_value):
    """Create a gauge chart for AQI display (adjusted for 90-155 range)"""
    if aqi_value is None:
        # Create a placeholder gauge for no data
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current AQI", 'font': {'size': 24, 'color': '#667eea'}},
            gauge={
                'axis': {'range': [90, 155]},
                'bar': {'color': "#808080"},
                'steps': [
                    {'range': [90, 100], 'color': "#00E400"},
                    {'range': [100, 120], 'color': "#FFA500"},
                    {'range': [120, 140], 'color': "#FF7E00"},
                    {'range': [140, 155], 'color': "#FF0000"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 155
                }
            }
        ))
        fig.add_annotation(
            text="No Data Available",
            x=0.5, y=0.3,
            showarrow=False,
            font=dict(size=18, color="gray")
        )
    else:
        try:
            aqi_value = float(aqi_value)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=aqi_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current AQI", 'font': {'size': 24, 'color': '#667eea'}},
                delta={'reference': 125, 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [90, 155]},
                    'bar': {'color': get_aqi_color(aqi_value)},
                    'steps': [
                        {'range': [90, 100], 'color': "#00E400"},
                        {'range': [100, 120], 'color': "#FFA500"},
                        {'range': [120, 140], 'color': "#FF7E00"},
                        {'range': [140, 155], 'color': "#FF0000"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 155
                    }
                }
            ))
        except (ValueError, TypeError):
            # Fallback for invalid values
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current AQI", 'font': {'size': 24, 'color': '#667eea'}},
                gauge={
                    'axis': {'range': [90, 155]},
                    'bar': {'color': "#808080"},
                    'steps': [
                        {'range': [90, 100], 'color': "#00E400"},
                        {'range': [100, 120], 'color': "#FFA500"},
                        {'range': [120, 140], 'color': "#FF7E00"},
                        {'range': [140, 155], 'color': "#FF0000"}
                    ]
                }
            ))
            fig.add_annotation(
                text="Invalid Data",
                x=0.5, y=0.3,
                showarrow=False,
                font=dict(size=18, color="red")
            )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_real_time_forecast_chart(forecast_data):
    """Create a real-time forecast chart with current time indicator"""
    if not forecast_data or 'forecast_data' not in forecast_data:
        return None
    
    forecast_list = forecast_data['forecast_data']
    if not forecast_list:
        return None
    
    df = pd.DataFrame(forecast_list)
    
    # Convert timestamp strings to datetime objects
    try:
        # More robust timestamp parsing
        parsed_timestamps = []
        for ts in df['timestamp']:
            try:
                if isinstance(ts, str):
                    # Handle string timestamps with timezone info
                    ts_clean = ts.replace('Z', '').replace('+00:00', '')
                    parsed_ts = pd.to_datetime(ts_clean)
                else:
                    parsed_ts = pd.to_datetime(ts)
                parsed_timestamps.append(parsed_ts)
            except Exception as parse_error:
                st.warning(f"Warning: Could not parse timestamp {ts}: {parse_error}")
                # Skip this timestamp
                continue
        
        if parsed_timestamps:
            df['timestamp'] = parsed_timestamps
            # Ensure we still have data after parsing
            if len(df) == 0:
                st.error("No valid forecast data remaining after timestamp parsing")
                return None
        else:
            st.error("No valid timestamps found in forecast data")
            return None
            
    except Exception as e:
        st.error(f"Error parsing timestamps: {e}")
        return None
    
    # Create the main forecast chart
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi_forecast'],
        mode='lines+markers',
        name='AQI Forecast',
        line=dict(color='#2E86AB', width=4, shape='spline'),
        marker=dict(size=8, color='#2E86AB', line=dict(width=2, color='white')),
        hovertemplate='<b>Time:</b> %{x}<br><b>AQI:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Add current time indicator - only if we have valid timestamps
    try:
        current_time = datetime.now()
        # Check if current time is within the forecast range
        if len(df['timestamp']) > 0:
            # Ensure all timestamps are properly converted to datetime
            valid_timestamps = []
            for ts in df['timestamp']:
                try:
                    if isinstance(ts, str):
                        # Handle string timestamps
                        ts_clean = ts.replace('Z', '').replace('+00:00', '')
                        valid_ts = pd.to_datetime(ts_clean)
                    else:
                        valid_ts = pd.to_datetime(ts)
                    valid_timestamps.append(valid_ts)
                except Exception:
                    # Skip invalid timestamps
                    continue
            
            if valid_timestamps:
                min_time = min(valid_timestamps)
                max_time = max(valid_timestamps)
                
                # Only add current time marker if it's within the forecast range
                if min_time <= current_time <= max_time:
                    fig.add_vline(
                        x=current_time,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text="Current Time",
                        annotation_position="top right"
                    )
    except Exception as e:
        # If there's any error with the current time marker, just continue without it
        st.warning(f"Could not add current time marker: {e}")
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi_forecast'] * 1.1,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(135, 206, 235, 0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['aqi_forecast'] * 0.9,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(135, 206, 235, 0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(135, 206, 235, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Professional styling
    fig.update_layout(
        title="🕐 Real-Time AQI Forecast: Next 72 Hours from Current Time (Red line = Current Time)",
        xaxis_title="Time",
        yaxis_title="AQI Value",
        height=500,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#2c3e50'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False, range=[85, 160])
    
    return fig

def get_current_aqi_from_forecast(forecast_data):
    """Get current AQI based on actual time from forecast data"""
    if not forecast_data or 'forecast_data' not in forecast_data:
        return None
    
    forecast_list = forecast_data['forecast_data']
    if not forecast_list:
        return None
    
    current_time = datetime.now()
    
    # Find the closest forecast to current time
    closest_forecast = None
    min_time_diff = float('inf')
    
    for forecast in forecast_list:
        try:
            # Handle different timestamp formats
            timestamp_str = forecast['timestamp']
            if isinstance(timestamp_str, str):
                # Remove timezone info if present and parse
                timestamp_str = timestamp_str.replace('Z', '').replace('+00:00', '')
                forecast_time = pd.to_datetime(timestamp_str)
            else:
                forecast_time = pd.to_datetime(timestamp_str)
            
            # Calculate time difference in seconds
            time_diff = abs((forecast_time - current_time).total_seconds())
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_forecast = forecast
                
        except Exception as e:
            # Skip this forecast if timestamp parsing fails
            continue
    
    # Return AQI if we found a close forecast (within 1 hour)
    if closest_forecast and min_time_diff < 3600:  # Within 1 hour
        try:
            return float(closest_forecast['aqi_forecast'])
        except (ValueError, TypeError):
            return None
    
    return None

def auto_data_pipeline():
    """Automatically run data collection and forecasting pipeline"""
    if not st.session_state.auto_mode:
        return
    
    # Check if we need to collect new data (every 10 minutes)
    if (st.session_state.last_data_collection is None or 
        (datetime.now() - st.session_state.last_data_collection).total_seconds() > 600):
        
        with st.spinner("🔄 Auto-collecting real-time data..."):
            collect_result = collect_real_time_data()
            if collect_result:
                st.success("✅ Auto-data collection completed!")
    
    # Check if we need to generate new forecast (every 30 minutes)
    if (st.session_state.last_forecast is None or 
        (datetime.now() - st.session_state.last_forecast).total_seconds() > 1800):
        
        with st.spinner("🔮 Auto-generating forecast..."):
            forecast_result = get_forecast()
            if forecast_result:
                st.session_state.forecast_data = forecast_result
                st.success("✅ Auto-forecast completed!")
    
    # Update current AQI from forecast
    if st.session_state.forecast_data:
        current_aqi = get_current_aqi_from_forecast(st.session_state.forecast_data)
        if current_aqi:
            st.session_state.current_aqi_data = {
                'current_aqi': current_aqi,
                'timestamp': datetime.now().isoformat(),
                'status': 'auto_forecast',
                'location': 'Peshawar, Pakistan',
                'last_update': 'Auto-updated from forecast',
                'data_available': True
            }

def show_real_time_dashboard():
    """Show the main real-time dashboard"""
    # Live status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="live-indicator">🟢 LIVE - Auto-Updating Every 2 Minutes</div>', unsafe_allow_html=True)
    
    # Current time and forecast period display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        current_time = datetime.now()
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; border-radius: 15px; margin: 1rem 0;">
            <h4>🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</h4>
            <p>📅 Forecast Period: Next 72 hours from current time</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.markdown("""
        <div class="warning-box">
            <h3>❌ Backend API Connection Error</h3>
            <p>Please check if the FastAPI server is running on port 8001</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Auto-mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auto_mode = st.checkbox("🤖 Enable Auto-Mode", value=st.session_state.auto_mode, key="auto_checkbox")
        st.session_state.auto_mode = auto_mode
        
        if auto_mode:
            st.info("🔄 Auto-mode enabled: Data collection and forecasting every 10-30 minutes")
        else:
            st.warning("⏸️ Auto-mode disabled: Manual control required")
    
    # Run auto pipeline
    if st.session_state.auto_mode:
        auto_data_pipeline()
    
    # Current AQI Section with real-time data
    st.header("📊 Live Air Quality Status")
    
    # Get current data (either from auto-forecast or API)
    current_data = st.session_state.current_aqi_data
    if not current_data:
        current_data = get_current_aqi()
        if current_data:
            st.session_state.current_aqi_data = current_data
    
    if current_data:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            aqi_value = current_data.get('current_aqi')
            if aqi_value is not None:
                data_status = current_data.get('status', 'success')
                if data_status == 'auto_forecast':
                    st.metric(
                        "Current AQI (Live)", 
                        f"{aqi_value:.1f}", 
                        f"Status: {get_aqi_category(aqi_value)}"
                    )
                else:
                    st.metric(
                        "Current AQI", 
                        f"{aqi_value:.1f}", 
                        f"Status: {get_aqi_category(aqi_value)}"
                    )
            else:
                st.metric("Current AQI", "N/A", "Status: Unknown")
        
        with col2:
            if aqi_value is not None:
                category = get_aqi_category(aqi_value)
                color_class = f"status-{category.lower().replace(' ', '-')}"
                st.markdown(f'<div class="metric-card"><h3>Air Quality</h3><p class="{color_class}">{category}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>Air Quality</h3><p class="status-warning">Unknown</p></div>', unsafe_allow_html=True)
        
        with col3:
            timestamp = current_data.get('timestamp', 'Unknown')
            st.metric("Last Update", timestamp, "Time: Recent")
        
        with col4:
            data_status = current_data.get('status', 'success')
            if data_status == 'auto_forecast':
                st.metric("Forecast Status", "Live", "Auto-updating")
            elif data_status == 'success':
                st.metric("Forecast Status", "Ready", "Models: 3/3 Active")
            else:
                st.metric("Forecast Status", "Limited", "Manual mode")
        
        # AQI Gauge with enhanced layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if aqi_value is not None:
                gauge_fig = create_aqi_gauge(aqi_value)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Show data source indicator
                data_status = current_data.get('status', 'success')
                if data_status == 'auto_forecast':
                    st.success("✅ **Live Forecast Mode**: Current AQI automatically calculated from real-time forecast data!")
                elif data_status == 'success':
                    st.success("✅ **Live Data**: Real-time AQI data from environmental sensors.")
                else:
                    st.info("🔄 **Demo Mode**: This is sample AQI data.")
            else:
                st.warning("⚠️ No AQI data available to display gauge")
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>📊 AQI Categories Range)</h3>
                <p>🟢 <strong>Good </strong> Air quality is satisfactory</p>
                <p>🟡 <strong>Moderate :</strong> Air quality is acceptable</p>
                <p>🟠 <strong>Unhealthy for Sensitive Groups :</strong> Some people may be affected</p>
                <p>🔴 <strong>Unhealthy </strong> Everyone may begin to experience health effects</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time forecast display
        if st.session_state.forecast_data and 'forecast_data' in st.session_state.forecast_data:
            st.subheader("🕐 Live Forecast Visualization")
            
            # Show current forecast chart
            try:
                # Validate forecast data before creating chart
                if not st.session_state.forecast_data.get('forecast_data'):
                    st.warning("⚠️ Forecast data is empty or malformed. Please regenerate the forecast.")
                    return
                
                forecast_chart = create_real_time_forecast_chart(st.session_state.forecast_data)
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                else:
                    st.warning("⚠️ Could not create forecast chart. Please regenerate the forecast.")
            except Exception as e:
                st.error(f"❌ Error creating forecast chart: {e}")
                st.info("Please try regenerating the forecast or check the data format.")
                # Show debug info
                with st.expander("🔍 Debug Information"):
                    st.write("Forecast data structure:")
                    st.json(st.session_state.forecast_data)
            
            # Forecast summary
            if 'forecast_summary' in st.session_state.forecast_data:
                summary = st.session_state.forecast_data['forecast_summary']
                
                # Calculate actual forecast time range
                if st.session_state.forecast_data['forecast_data']:
                    forecast_list = st.session_state.forecast_data['forecast_data']
                    if forecast_list:
                        try:
                            first_timestamp = pd.to_datetime(forecast_list[0]['timestamp'])
                            last_timestamp = pd.to_datetime(forecast_list[-1]['timestamp'])
                            time_range = f"{first_timestamp.strftime('%Y-%m-%d %H:%M')} to {last_timestamp.strftime('%Y-%m-%d %H:%M')}"
                        except:
                            time_range = "Next 72 hours"
                    else:
                        time_range = "Next 72 hours"
                else:
                    time_range = "Next 72 hours"
                
                st.markdown(f"""
                <div class="forecast-card">
                    <h3>📈 Live Forecast Summary</h3>
                    <p><strong>Period:</strong> {summary.get('period', 'Unknown')}</p>
                    <p><strong>Time Range:</strong> {time_range}</p>
                    <p><strong>Predictions:</strong> {summary.get('predictions', 0)}</p>
                    <p><strong>AQI Range:</strong> {summary.get('aqi_range', 'Unknown')}</p>
                    <p><strong>Last Updated:</strong> {st.session_state.last_forecast.strftime('%H:%M:%S') if st.session_state.last_forecast else 'Never'}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No forecast data available yet. Click 'Generate Forecast' to create a 72-hour forecast starting from the current time!")
        
        # Action buttons
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📊 Collect Data", use_container_width=True):
                with st.spinner("Collecting real-time data..."):
                    result = collect_real_time_data()
                    if result:
                        st.success("✅ Data collected successfully!")
        
        with col3:
            # Show forecast button with status indicator
            if st.button("🔮 Generate Forecast", use_container_width=True):
                if st.session_state.forecast_data:
                    st.info("📊 Forecast already available! Click 'Refresh Now' to update.")
                else:
                    get_forecast()
        
        with col4:
            if st.button("📈 View Analytics", use_container_width=True):
                st.session_state.page = "Analytics"
                st.rerun()
        
        # Model Status Indicator
        st.subheader("🤖 Model Status")
        
        # Check if we have forecast data (indicates models are ready)
        if st.session_state.forecast_data:
            st.success("✅ **Models Ready**: All forecasting models are trained and ready")
            st.info("💡 **Tip**: Models are now cached and subsequent forecasts will be much faster!")
        else:
            st.warning("⚠️ **Models Not Ready**: Models need to be trained before forecasting")
            st.info("🔧 **Next Step**: Click 'Generate Forecast' to train models (first run takes 1-2 minutes)")
        
        # Performance Tips
        with st.expander("💡 Performance Tips"):
            st.write("**First Run (Model Training):**")
            st.write("• Takes 1-2 minutes to train all models")
            st.write("• Models are saved for future use")
            st.write("• Historical data is processed once")
            
            st.write("**Subsequent Runs:**")
            st.write("• Forecasts generate in 10-30 seconds")
            st.write("• Models are loaded from disk")
            st.write("• No retraining required")
            
            st.write("**If Timeout Occurs:**")
            st.write("• Wait 2-3 minutes for first training to complete")
            st.write("• Check backend API console for progress")
            st.write("• Ensure sufficient system resources")
    
    else:
        st.error("❌ Failed to retrieve current AQI data")

def show_analytics():
    """Show analytics page with real-time data insights"""
    if not st.session_state.forecast_data:
        st.warning("⚠️ No forecast data available. Generate a forecast first to see analytics.")
        return
    
    # Professional color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D'
    }
    
    # Feature Importance Analysis - Use real data from forecast
    st.subheader("🔍 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use real feature importance if available, otherwise show message
        if 'feature_importance' in st.session_state.forecast_data:
            # Extract real feature importance data
            feature_data = st.session_state.forecast_data['feature_importance']
            if isinstance(feature_data, dict) and feature_data:
                features = list(feature_data.keys())
                importance_scores = list(feature_data.values())
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_scores,
                    y=features,
                    orientation='h',
                    marker_color=[colors['success'] if score > 0.8 else colors['accent'] if score > 0.7 else colors['secondary'] for score in importance_scores],
                    text=[f'{score:.2f}' for score in importance_scores],
                    textposition='auto',
                    textfont=dict(size=12, color='white')
                ))
                
                fig.update_layout(
                    title="Real Feature Importance from Forecast Models",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Feature importance data will be available after generating a forecast with feature analysis enabled.")
        else:
            st.info("📊 Feature importance analysis will be available after generating a forecast. This shows which environmental factors most influence AQI predictions.")
    
    with col2:
        # Model Performance - Use real data from forecast
        if 'model_performance' in st.session_state.forecast_data:
            models = list(st.session_state.forecast_data['model_performance'].keys())
            scores = list(st.session_state.forecast_data['model_performance'].values())
            
            model_names = {
                'random_forest': '🌳 Random Forest',
                'gradient_boosting': '🚀 Gradient Boosting', 
                'lstm': '🧠 LSTM Neural Network'
            }
            
            display_names = [model_names.get(model, model.title()) for model in models]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=display_names,
                y=scores,
                marker_color=[colors['success'] if score > 0.9 else colors['accent'] if score > 0.8 else colors['secondary'] for score in scores],
                text=[f'{score:.3f}' for score in scores],
                textposition='outside',
                textfont=dict(size=12, color='white')
            ))
            
            fig.update_layout(
                title="Real Model Performance (R² Scores)",
                xaxis_title="Models",
                yaxis_title="R² Score",
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Model performance data will be available after generating a forecast. This shows how well each ML model performed during training.")
    
    # Real-time forecast trends - Use real forecast data
    if st.session_state.forecast_data and 'forecast_data' in st.session_state.forecast_data:
        st.subheader("📈 Real-Time Forecast Trends")
        
        # Add forecast timing information
        st.info("🕐 **Forecast Period**: This chart shows AQI predictions for the next 72 hours starting from the current time.")
        
        forecast_list = st.session_state.forecast_data['forecast_data']
        if not forecast_list:
            st.warning("⚠️ No forecast data available for trend analysis.")
            return
            
        df = pd.DataFrame(forecast_list)
        
        try:
            # More robust timestamp parsing
            parsed_timestamps = []
            for ts in df['timestamp']:
                try:
                    if isinstance(ts, str):
                        # Handle string timestamps with timezone info
                        ts_clean = ts.replace('Z', '').replace('+00:00', '')
                        parsed_ts = pd.to_datetime(ts_clean)
                    else:
                        parsed_ts = pd.to_datetime(ts)
                    parsed_timestamps.append(parsed_ts)
                except Exception as parse_error:
                    st.warning(f"Warning: Could not parse timestamp {ts}: {parse_error}")
                    # Skip this timestamp
                    continue
            
            if parsed_timestamps:
                df['timestamp'] = parsed_timestamps
                # Ensure we still have data after parsing
                if len(df) == 0:
                    st.error("No valid forecast data remaining after timestamp parsing")
                    return
            else:
                st.error("No valid timestamps found in forecast data")
                return
            
            # Create trend analysis using real data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['aqi_forecast'],
                mode='lines+markers',
                name='AQI Forecast',
                line=dict(color=colors['primary'], width=3),
                marker=dict(size=6, color=colors['primary'])
            ))
            
            # Add current time marker - only if within range
            current_time = datetime.now()
            if len(df['timestamp']) > 0:
                # Ensure all timestamps are properly converted to datetime
                valid_timestamps = []
                for ts in df['timestamp']:
                    try:
                        if isinstance(ts, str):
                            # Handle string timestamps
                            ts_clean = ts.replace('Z', '').replace('+00:00', '')
                            valid_ts = pd.to_datetime(ts_clean)
                        else:
                            valid_ts = pd.to_datetime(ts)
                        valid_timestamps.append(valid_ts)
                    except Exception:
                        # Skip invalid timestamps
                        continue
                
                if valid_timestamps:
                    min_time = min(valid_timestamps)
                    max_time = max(valid_timestamps)
                    
                    if min_time <= current_time <= max_time:
                        fig.add_vline(
                            x=current_time,
                            line_dash="dash",
                            line_color="red",
                            line_width=3,
                            annotation_text="Current Time"
                        )
            
            fig.update_layout(
                title="Real-Time AQI Forecast Trend: Next 72 Hours from Current Time",
                xaxis_title="Time",
                yaxis_title="AQI Value",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating trend chart: {e}")
            st.info("Forecast data format issue. Please regenerate the forecast.")
    else:
        st.info("📊 Forecast trend analysis will be available after generating a forecast. This shows the predicted AQI changes over the next 72 hours.")
    
    # Real-time Data Summary - Show actual statistics from forecast
    if st.session_state.forecast_data and 'forecast_data' in st.session_state.forecast_data:
        st.subheader("📊 Real-Time Data Summary")
        
        forecast_list = st.session_state.forecast_data['forecast_data']
        if forecast_list:
            try:
                # Calculate real statistics from forecast data
                aqi_values = [float(item.get('aqi_forecast', 0)) for item in forecast_list if item.get('aqi_forecast') is not None]
                
                if aqi_values:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Average AQI",
                            f"{sum(aqi_values) / len(aqi_values):.1f}",
                            "Predicted average"
                        )
                    
                    with col2:
                        st.metric(
                            "Min AQI",
                            f"{min(aqi_values):.1f}",
                            "Lowest prediction"
                        )
                    
                    with col3:
                        st.metric(
                            "Max AQI",
                            f"{max(aqi_values):.1f}",
                            "Highest prediction"
                        )
                    
                    with col4:
                        st.metric(
                            "Total Hours",
                            len(forecast_list),
                            "Forecast coverage"
                        )
                    
                    # Show AQI distribution
                    st.subheader("📈 AQI Distribution Analysis")
                    
                    # Create histogram of AQI values
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=aqi_values,
                        nbinsx=20,
                        marker_color=colors['primary'],
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title="Distribution of Predicted AQI Values",
                        xaxis_title="AQI Value",
                        yaxis_title="Frequency",
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show time-based patterns
                    if len(forecast_list) > 0:
                        st.subheader("🕐 Time-Based AQI Patterns")
                        
                        # Group by hour of day to show patterns
                        hourly_data = {}
                        for item in forecast_list:
                            try:
                                timestamp = pd.to_datetime(item['timestamp'])
                                hour = timestamp.hour
                                aqi = float(item.get('aqi_forecast', 0))
                                
                                if hour not in hourly_data:
                                    hourly_data[hour] = []
                                hourly_data[hour].append(aqi)
                            except:
                                continue
                        
                        if hourly_data:
                            hours = sorted(hourly_data.keys())
                            avg_aqi_by_hour = [sum(hourly_data[hour]) / len(hourly_data[hour]) for hour in hours]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=hours,
                                y=avg_aqi_by_hour,
                                mode='lines+markers',
                                name='Average AQI by Hour',
                                line=dict(color=colors['accent'], width=3),
                                marker=dict(size=8, color=colors['accent'])
                            ))
                            
                            fig.update_layout(
                                title="Average AQI by Hour of Day (Next 72 Hours)",
                                xaxis_title="Hour of Day (0-23)",
                                yaxis_title="Average AQI",
                                height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating data summary: {e}")
                st.info("Please regenerate the forecast to see data summary.")
        else:
            st.warning("⚠️ No forecast data available for summary analysis.")
    else:
        st.info("📊 Data summary will be available after generating a forecast. This shows real statistics and patterns from your predictions.")
    
    # Additional Trend Analysis Graphs
    if st.session_state.forecast_data and 'forecast_data' in st.session_state.forecast_data:
        st.subheader("📊 Additional Trend Analysis")
        
        forecast_list = st.session_state.forecast_data['forecast_data']
        if forecast_list:
            try:
                df = pd.DataFrame(forecast_list)
                
                # Robust timestamp parsing for new charts
                parsed_timestamps = []
                for ts in df['timestamp']:
                    try:
                        if isinstance(ts, str):
                            ts_clean = ts.replace('Z', '').replace('+00:00', '')
                            parsed_ts = pd.to_datetime(ts_clean)
                        else:
                            parsed_ts = pd.to_datetime(ts)
                        parsed_timestamps.append(parsed_ts)
                    except Exception:
                        continue
                
                if parsed_timestamps:
                    df['timestamp'] = parsed_timestamps
                    
                    # Chart 1: AQI Trend with Confidence Bands
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = go.Figure()
                        
                        # Main AQI trend
                        fig1.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df['aqi_forecast'],
                            mode='lines+markers',
                            name='AQI Forecast',
                            line=dict(color=colors['primary'], width=3),
                            marker=dict(size=6, color=colors['primary'])
                        ))
                        
                        # Add confidence bands if available
                        if 'confidence' in df.columns:
                            confidence_values = df['confidence'].fillna(0.5)
                            upper_bound = df['aqi_forecast'] + (confidence_values * 10)
                            lower_bound = df['aqi_forecast'] - (confidence_values * 10)
                            
                            fig1.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=upper_bound,
                                mode='lines',
                                name='Upper Confidence',
                                line=dict(color='rgba(46, 134, 171, 0.3)', width=1),
                                showlegend=False
                            ))
                            
                            fig1.add_trace(go.Scatter(
                                x=df['timestamp'],
                                y=lower_bound,
                                mode='lines',
                                name='Lower Confidence',
                                line=dict(color='rgba(46, 134, 171, 0.3)', width=1),
                                fill='tonexty',
                                fillcolor='rgba(46, 134, 171, 0.1)',
                                showlegend=False
                            ))
                        
                        fig1.update_layout(
                            title="AQI Forecast with Confidence Bands",
                            xaxis_title="Time",
                            yaxis_title="AQI Value",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    # Chart 2: AQI Change Rate Analysis
                    with col2:
                        # Calculate hourly change rate
                        df_sorted = df.sort_values('timestamp')
                        df_sorted['aqi_change'] = df_sorted['aqi_forecast'].diff()
                        df_sorted['hourly_change_rate'] = df_sorted['aqi_change'] / 1  # Change per hour
                        
                        fig2 = go.Figure()
                        
                        # Positive changes (improving air quality)
                        positive_changes = df_sorted[df_sorted['hourly_change_rate'] > 0]
                        if len(positive_changes) > 0:
                            fig2.add_trace(go.Scatter(
                                x=positive_changes['timestamp'],
                                y=positive_changes['hourly_change_rate'],
                                mode='markers',
                                name='Improving AQI',
                                marker=dict(color='green', size=8, symbol='circle'),
                                hovertemplate='Time: %{x}<br>Improvement Rate: %{y:.2f}/hr<extra></extra>'
                            ))
                        
                        # Negative changes (worsening air quality)
                        negative_changes = df_sorted[df_sorted['hourly_change_rate'] < 0]
                        if len(negative_changes) > 0:
                            fig2.add_trace(go.Scatter(
                                x=negative_changes['timestamp'],
                                y=negative_changes['hourly_change_rate'],
                                mode='markers',
                                name='Worsening AQI',
                                marker=dict(color='red', size=8, symbol='circle'),
                                hovertemplate='Time: %{x}<br>Deterioration Rate: %{y:.2f}/hr<extra></extra>'
                            ))
                        
                        # Add zero line for reference
                        fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                        
                        fig2.update_layout(
                            title="AQI Change Rate Analysis (Hourly)",
                            xaxis_title="Time",
                            yaxis_title="AQI Change Rate (/hour)",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Chart 3: AQI Trend Comparison (First vs Last 24 hours)
                    st.subheader("🔄 AQI Trend Comparison")
                    
                    if len(df) >= 48:  # Need at least 48 hours for comparison
                        first_24h = df.head(24)
                        last_24h = df.tail(24)
                        
                        fig3 = go.Figure()
                        
                        # First 24 hours
                        fig3.add_trace(go.Scatter(
                            x=first_24h['timestamp'],
                            y=first_24h['aqi_forecast'],
                            mode='lines+markers',
                            name='First 24 Hours',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6, color='blue')
                        ))
                        
                        # Last 24 hours
                        fig3.add_trace(go.Scatter(
                            x=last_24h['timestamp'],
                            y=last_24h['aqi_forecast'],
                            mode='lines+markers',
                            name='Last 24 Hours',
                            line=dict(color='orange', width=3),
                            marker=dict(size=6, color='orange')
                        ))
                        
                        fig3.update_layout(
                            title="AQI Trend Comparison: First vs Last 24 Hours",
                            xaxis_title="Time",
                            yaxis_title="AQI Value",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Show trend statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            first_avg = first_24h['aqi_forecast'].mean()
                            last_avg = last_24h['aqi_forecast'].mean()
                            trend_direction = "📈 Improving" if last_avg < first_avg else "📉 Worsening" if last_avg > first_avg else "➡️ Stable"
                            
                            st.metric(
                                "Trend Direction",
                                trend_direction,
                                f"Change: {last_avg - first_avg:.1f}"
                            )
                        
                        with col2:
                            first_std = first_24h['aqi_forecast'].std()
                            last_std = last_24h['aqi_forecast'].std()
                            volatility_change = "📊 More Variable" if last_std > first_std else "📊 Less Variable" if last_std < first_std else "📊 Same Variability"
                            
                            st.metric(
                                "Volatility Change",
                                volatility_change,
                                f"Std: {first_std:.1f} → {last_std:.1f}"
                            )
                    else:
                        st.info("📊 Need at least 48 hours of forecast data for trend comparison analysis.")
                
            except Exception as e:
                st.error(f"Error creating additional trend charts: {e}")
                st.info("Please regenerate the forecast to see additional trend analysis.")
        else:
            st.warning("⚠️ No forecast data available for additional trend analysis.")
    else:
        st.info("📊 Additional trend analysis will be available after generating a forecast.")

def show_historical_eda():
    """Show comprehensive historical EDA using the actual historical_merged.csv file"""
    
    # Set pandas to handle PyArrow compatibility issues
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')
    
    try:
        # Load historical data
        try:
            csv_path = "data_repositories/historical_data/processed/historical_merged.csv"
            if os.path.exists(csv_path):
                # Try to load with PyArrow engine first, fallback to default if it fails
                try:
                    df = pd.read_csv(csv_path, engine='pyarrow')
                    st.success(f"✅ Successfully loaded historical data with PyArrow: {len(df)} records")
                except Exception as pyarrow_error:
                    st.warning(f"⚠️ PyArrow engine failed, using default engine: {pyarrow_error}")
                    df = pd.read_csv(csv_path)
                    st.success(f"✅ Successfully loaded historical data with default engine: {len(df)} records")
                
                # Validate required columns - check for timestamp column with different possible names
                timestamp_cols = ['timestamp', 'date', 'time', 'datetime']
                found_timestamp_col = None
                for col in timestamp_cols:
                    if col in df.columns:
                        found_timestamp_col = col
                        break
                
                if found_timestamp_col is None:
                    st.error(f"❌ No timestamp column found. Available columns: {list(df.columns)}")
                    st.info("Please ensure the CSV file contains a timestamp column.")
                    return
                
                # Rename timestamp column to 'timestamp' for consistency
                if found_timestamp_col != 'timestamp':
                    df = df.rename(columns={found_timestamp_col: 'timestamp'})
                    st.info(f"📝 Renamed '{found_timestamp_col}' column to 'timestamp' for consistency")
                
                # Convert timestamp column to datetime for proper analysis
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.success("✅ Timestamp column converted to datetime format")
                except Exception as e:
                    st.warning(f"⚠️ Could not convert timestamp to datetime: {e}")
                    st.info("Some time-based analysis features may not work properly")
                
                # Store in session state for reuse
                st.session_state.historical_data = df
                
            else:
                st.error(f"❌ Historical data file not found: {csv_path}")
                st.info("Please ensure the file exists at the specified path.")
                return
                
        except Exception as e:
            st.error(f"❌ Error loading historical data: {e}")
            return
        
        # Data Overview
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            # Convert timestamp to string format for display
            try:
                min_date = str(df['timestamp'].min())[:10]
                max_date = str(df['timestamp'].max())[:10]
                st.metric("Date Range", f"{min_date} to {max_date}")
            except Exception as e:
                st.metric("Date Range", "N/A")
                st.warning(f"Could not parse date range: {e}")
        with col3:
            st.metric("Columns", len(df.columns))
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Show data structure
        with st.expander("🔍 Data Structure & Info"):
            st.write("**Data Types:**")
            st.write(df.dtypes)
            
            # Convert datetime columns to strings to avoid PyArrow serialization issues
            display_df = df.copy()
            
            # Convert all datetime-like columns to strings to prevent PyArrow errors
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    try:
                        # Check if it's a datetime column by trying to parse first few values
                        sample_values = display_df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            pd.to_datetime(sample_values.iloc[0])
                            # It's a datetime column, convert to string
                            display_df[col] = display_df[col].astype(str)
                    except:
                        # Not a datetime column, leave as is
                        pass
            
            st.write("**First 5 Records:**")
            try:
                # Additional safety: ensure all columns are string-compatible
                safe_display_df = display_df.head().copy()
                for col in safe_display_df.columns:
                    if safe_display_df[col].dtype == 'object':
                        safe_display_df[col] = safe_display_df[col].astype(str)
                
                st.dataframe(safe_display_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying dataframe: {e}")
                st.write("**Data preview (first 3 rows):**")
                st.write(display_df.head(3).to_dict('records'))
            
            st.write("**Last 5 Records:**")
            try:
                # Additional safety: ensure all columns are string-compatible
                safe_display_df = display_df.tail().copy()
                for col in safe_display_df.columns:
                    if safe_display_df[col].dtype == 'object':
                        safe_display_df[col] = safe_display_df[col].astype(str)
                
                st.dataframe(safe_display_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying dataframe: {e}")
                st.write("**Data preview (last 3 rows):**")
                st.write(display_df.tail(3).to_dict('records'))
        
        # Column analysis
        st.subheader("📊 Column Analysis")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Columns:**")
            for col in numeric_cols:
                st.write(f"• {col}")
        
        with col2:
            st.write("**Categorical Columns:**")
            for col in categorical_cols:
                st.write(f"• {col}")
        
        # Time series analysis
        st.subheader("🕐 Time Series Analysis")
        
        try:
            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate AQI from pollutants if not present
            if 'aqi' not in df.columns:
                st.info("📊 **Note**: AQI column not found. Calculating AQI from pollutant concentrations using regional standards.")
                
                # Calculate AQI from PM2.5 and PM10 (simplified regional calculation)
                def calculate_regional_aqi(row):
                    try:
                        pm25 = row.get('pm2_5', 0)
                        pm10 = row.get('pm10', 0)
                        
                        if pd.isna(pm25) or pd.isna(pm10):
                            return 120.0  # Default value
                        
                        # Regional AQI calculation for Peshawar (90-155 range)
                        if pm25 <= 15.0:
                            aqi_pm25 = 90 + 20 * (pm25 / 15.0)  # 90-110 range
                        elif pm25 <= 25.0:
                            aqi_pm25 = 110 + 20 * ((pm25 - 15.0) / (25.0 - 15.0))  # 110-130 range
                        elif pm25 <= 35.0:
                            aqi_pm25 = 130 + 15 * ((pm25 - 25.0) / (35.0 - 25.0))  # 130-145 range
                        else:
                            aqi_pm25 = 145 + 10 * ((pm25 - 35.0) / (45.0 - 35.0))  # 145-155 range
                        
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
                        
                    except:
                        return 120.0
                
                # Apply AQI calculation
                df['aqi'] = df.apply(calculate_regional_aqi, axis=1)
                st.success("✅ AQI calculated from pollutant data using regional standards")
            
            # Safety check: ensure AQI column exists and has valid data
            if 'aqi' not in df.columns or df['aqi'].isnull().all():
                st.error("❌ AQI column is missing or contains no valid data")
                st.info("Cannot proceed with time series analysis without AQI data.")
                return
            
            # Safety check: ensure timestamp column is datetime for time-based analysis
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                st.warning("⚠️ Timestamp column is not in datetime format. Converting...")
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.success("✅ Timestamp converted to datetime format")
                except Exception as e:
                    st.error(f"❌ Cannot convert timestamp to datetime: {e}")
                    st.info("Time-based analysis features will be disabled")
                    return
            
            # Time-based visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily AQI trends
                daily_aqi = df.groupby(df['timestamp'].dt.date)['aqi'].mean().reset_index()
                daily_aqi['timestamp'] = pd.to_datetime(daily_aqi['timestamp'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_aqi['timestamp'],
                    y=daily_aqi['aqi'],
                    mode='lines+markers',
                    name='Daily Average AQI',
                    line=dict(color='#2E86AB', width=3)
                ))
                
                fig.update_layout(
                    title="Daily Average AQI Over Time",
                    xaxis_title="Date",
                    yaxis_title="Average AQI",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly AQI patterns
                hourly_aqi = df.groupby(df['timestamp'].dt.hour)['aqi'].mean().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly_aqi['timestamp'],
                    y=hourly_aqi['aqi'],
                    name='Hourly Average AQI',
                    marker_color='#A23B72'
                ))
                
                fig.update_layout(
                    title="Hourly AQI Patterns (24-Hour Cycle)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Average AQI",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly and seasonal patterns
            st.subheader("📅 Monthly & Seasonal Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly AQI trends
                monthly_aqi = df.groupby(df['timestamp'].dt.month)['aqi'].mean().reset_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_aqi['month_name'] = [month_names[i-1] for i in monthly_aqi['timestamp']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_aqi['month_name'],
                    y=monthly_aqi['aqi'],
                    name='Monthly Average AQI',
                    marker_color='#F18F01'
                ))
                
                fig.update_layout(
                    title="Monthly AQI Patterns",
                    xaxis_title="Month",
                    yaxis_title="Average AQI",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Day of week patterns
                df['day_of_week'] = df['timestamp'].dt.day_name()
                dow_aqi = df.groupby('day_of_week')['aqi'].mean().reset_index()
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_aqi = dow_aqi.set_index('day_of_week').reindex(dow_order).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dow_aqi['day_of_week'],
                    y=dow_aqi['aqi'],
                    name='Day of Week AQI',
                    marker_color='#C73E1D'
                ))
                
                fig.update_layout(
                    title="Day of Week AQI Patterns",
                    xaxis_title="Day of Week",
                    yaxis_title="Average AQI",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in time series analysis: {e}")
        
        # Enhanced Distribution Analysis
        st.subheader("📈 Enhanced Distribution Analysis")
        
        try:
            # PM2.5 & PM10 Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                if 'pm2_5' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df['pm2_5'],
                        nbinsx=30,
                        marker_color='crimson',
                        opacity=0.7,
                        name='PM2.5'
                    ))
                    
                    fig.update_layout(
                        title="PM2.5 Distribution",
                        xaxis_title="PM2.5 Concentration (μg/m³)",
                        yaxis_title="Frequency",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("PM2.5 data not available")
            
            with col2:
                if 'pm10' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df['pm10'],
                        nbinsx=30,
                        marker_color='navy',
                        opacity=0.7,
                        name='PM10'
                    ))
                    
                    fig.update_layout(
                        title="PM10 Distribution",
                        xaxis_title="PM10 Concentration (μg/m³)",
                        yaxis_title="Frequency",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("PM10 data not available")
            
            # AQI Distribution with KDE
            st.subheader("🌫️ AQI Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['aqi'],
                    nbinsx=30,
                    marker_color='#2E86AB',
                    opacity=0.7,
                    name='AQI Distribution'
                ))
                
                fig.update_layout(
                    title="AQI Distribution with Histogram",
                    xaxis_title="AQI Value",
                    yaxis_title="Frequency",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot for key pollutants
                pollutant_cols = ['pm2_5', 'pm10', 'no2', 'o3']
                available_pollutants = [col for col in pollutant_cols if col in df.columns]
                
                if available_pollutants:
                    fig = go.Figure()
                    for pollutant in available_pollutants:
                        fig.add_trace(go.Box(
                            y=df[pollutant],
                            name=pollutant.upper(),
                            boxpoints='outliers'
                        ))
                    
                    fig.update_layout(
                        title="Pollutant Distributions (Box Plots)",
                        yaxis_title="Concentration",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No pollutant data available for distribution analysis")
                    
        except Exception as e:
            st.error(f"Error in distribution analysis: {e}")
        
        # Advanced Pollutant Analysis
        st.subheader("🌫️ Advanced Pollutant Analysis")
        
        try:
            pollutant_cols = ['pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co', 'nh3']
            available_pollutants = [col for col in pollutant_cols if col in df.columns]
            
            if available_pollutants:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pollutant trends over time
                    fig = go.Figure()
                    
                    for pollutant in available_pollutants[:3]:  # Show first 3 pollutants
                        if pollutant in df.columns:
                            # Calculate daily averages
                            daily_pollutant = df.groupby(df['timestamp'].dt.date)[pollutant].mean().reset_index()
                            daily_pollutant['timestamp'] = pd.to_datetime(daily_pollutant['timestamp'])
                            
                            fig.add_trace(go.Scatter(
                                x=daily_pollutant['timestamp'],
                                y=daily_pollutant[pollutant],
                                mode='lines',
                                name=pollutant.upper(),
                                line=dict(width=2)
                            ))
                    
                    fig.update_layout(
                        title="Daily Pollutant Trends",
                        xaxis_title="Date",
                        yaxis_title="Concentration",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pollutant correlation heatmap
                    pollutant_data = df[available_pollutants].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=pollutant_data.values,
                        x=pollutant_data.columns,
                        y=pollutant_data.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=pollutant_data.values.round(2),
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title="Pollutant Correlation Matrix",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Pollutant statistics summary
                st.subheader("📊 Pollutant Statistics Summary")
                
                pollutant_stats = df[available_pollutants].describe()
                
                try:
                    st.dataframe(pollutant_stats.round(2), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying pollutant statistics: {e}")
                    # Fallback: display as formatted text
                    st.write("**Key Pollutant Statistics:**")
                    for pollutant in available_pollutants[:5]:  # Show first 5 pollutants
                        if pollutant in pollutant_stats.columns:
                            st.write(f"**{pollutant.upper()}:**")
                            st.write(f"  - Mean: {pollutant_stats[pollutant]['mean']:.2f}")
                            st.write(f"  - Min: {pollutant_stats[pollutant]['min']:.2f}")
                            st.write(f"  - Max: {pollutant_stats[pollutant]['max']:.2f}")
                            st.write("---")
                    
            else:
                st.info("No pollutant data available for analysis")
                
        except Exception as e:
            st.error(f"Error in pollutant analysis: {e}")
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        try:
            # Select numeric columns for correlation
            correlation_cols = ['aqi', 'pm2_5', 'pm10', 'no2', 'o3', 'temperature', 'relative_humidity', 'wind_speed']
            available_cols = [col for col in correlation_cols if col in df.columns]
            
            if len(available_cols) > 1:
                corr_data = df[available_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_data.values.round(2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlations
                st.write("**Top Correlations with AQI:**")
                aqi_corr = corr_data['aqi'].sort_values(ascending=False)
                for feature, corr in aqi_corr.items():
                    if feature != 'aqi':
                        st.write(f"• {feature}: {corr:.3f}")
            else:
                st.warning("⚠️ Insufficient numeric columns for correlation analysis")
                
        except Exception as e:
            st.error(f"Error in correlation analysis: {e}")
        
        # Additional Insightful EDA Graphs
        st.subheader("🔍 Additional Insights & Patterns")
        
        try:
            # Check if we have the required data
            if 'timestamp' not in df.columns:
                st.warning("⚠️ Timestamp column not found. Skipping time-based analysis.")
                return
            
            if 'aqi' not in df.columns:
                st.warning("⚠️ AQI column not found. Skipping AQI-based analysis.")
                return
            
            # Ensure timestamp is datetime
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                st.error(f"❌ Error converting timestamp: {e}")
                return
            
            # Seasonal AQI patterns
            st.subheader("🌤️ Seasonal AQI Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Seasonal AQI box plot
                    df['season'] = df['timestamp'].dt.month.map({
                        12: 'Winter', 1: 'Winter', 2: 'Winter',
                        3: 'Spring', 4: 'Spring', 5: 'Spring',
                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                        9: 'Fall', 10: 'Fall', 11: 'Fall'
                    })
                    
                    seasonal_aqi = df.groupby('season')['aqi'].apply(list).reset_index()
                    
                    fig = go.Figure()
                    for season in seasonal_aqi['season']:
                        season_data = seasonal_aqi[seasonal_aqi['season'] == season]['aqi'].iloc[0]
                        fig.add_trace(go.Box(
                            y=season_data,
                            name=season,
                            boxpoints='outliers'
                        ))
                    
                    fig.update_layout(
                        title="Seasonal AQI Patterns",
                        yaxis_title="AQI Value",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in seasonal analysis: {e}")
            
            with col2:
                try:
                    # PM2.5/PM10 ratio analysis
                    if 'pm2_5' in df.columns and 'pm10' in df.columns:
                        # Filter out invalid values
                        valid_data = df[(df['pm2_5'] > 0) & (df['pm10'] > 0)].copy()
                        
                        if len(valid_data) > 0:
                            valid_data['pm_ratio'] = valid_data['pm2_5'] / valid_data['pm10']
                            valid_data['pm_ratio'] = valid_data['pm_ratio'].replace([np.inf, -np.inf], np.nan)
                            valid_ratios = valid_data['pm_ratio'].dropna()
                            
                            if len(valid_ratios) > 0:
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=valid_ratios,
                                    nbinsx=25,
                                    marker_color='purple',
                                    opacity=0.7,
                                    name='PM2.5/PM10 Ratio'
                                ))
                                
                                fig.update_layout(
                                    title="PM2.5/PM10 Ratio Distribution",
                                    xaxis_title="Ratio Value",
                                    yaxis_title="Frequency",
                                    height=400,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show ratio statistics
                                ratio_stats = valid_ratios.describe()
                                st.write("**PM2.5/PM10 Ratio Statistics:**")
                                st.write(f"• Mean: {ratio_stats['mean']:.3f}")
                                st.write(f"• Median: {ratio_stats['50%']:.3f}")
                                st.write(f"• Range: {ratio_stats['min']:.3f} - {ratio_stats['max']:.3f}")
                            else:
                                st.info("No valid PM2.5/PM10 ratios available")
                        else:
                            st.info("No valid PM2.5/PM10 data for ratio analysis")
                    else:
                        st.info("PM2.5/PM10 ratio analysis not available")
                except Exception as e:
                    st.error(f"Error in PM ratio analysis: {e}")
            
            # Trend analysis
            st.subheader("📈 Trend Analysis & Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Rolling average AQI
                    if 'aqi' in df.columns:
                        df_sorted = df.sort_values('timestamp').copy()
                        
                        # Calculate rolling averages with error handling
                        try:
                            df_sorted['aqi_rolling_7d'] = df_sorted['aqi'].rolling(window=168, min_periods=1).mean()
                            df_sorted['aqi_rolling_24h'] = df_sorted['aqi'].rolling(window=24, min_periods=1).mean()
                        except Exception as e:
                            st.warning(f"⚠️ Rolling average calculation failed: {e}")
                            df_sorted['aqi_rolling_7d'] = df_sorted['aqi']
                            df_sorted['aqi_rolling_24h'] = df_sorted['aqi']
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_sorted['timestamp'],
                            y=df_sorted['aqi'],
                            mode='lines',
                            name='Hourly AQI',
                            line=dict(color='lightblue', width=1),
                            opacity=0.6
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_sorted['timestamp'],
                            y=df_sorted['aqi_rolling_24h'],
                            mode='lines',
                            name='24-Hour Rolling Average',
                            line=dict(color='orange', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_sorted['timestamp'],
                            y=df_sorted['aqi_rolling_7d'],
                            mode='lines',
                            name='7-Day Rolling Average',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig.update_layout(
                            title="AQI Trends with Rolling Averages",
                            xaxis_title="Time",
                            yaxis_title="AQI Value",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("AQI trend analysis not available")
                except Exception as e:
                    st.error(f"Error in trend analysis: {e}")
            
            with col2:
                try:
                    # Peak hour analysis
                    if 'aqi' in df.columns:
                        peak_hours = df.groupby(df['timestamp'].dt.hour)['aqi'].agg(['mean', 'std', 'count']).reset_index()
                        
                        # Filter out hours with insufficient data
                        peak_hours = peak_hours[peak_hours['count'] >= 5]  # At least 5 data points
                        
                        if len(peak_hours) > 0:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=peak_hours['timestamp'],
                                y=peak_hours['mean'],
                                mode='lines+markers',
                                name='Average AQI',
                                line=dict(color='green', width=3)
                            ))
                            
                            # Add error bars only if std is available
                            if 'std' in peak_hours.columns:
                                fig.add_trace(go.Scatter(
                                    x=peak_hours['timestamp'],
                                    y=peak_hours['mean'] + peak_hours['std'],
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(color='rgba(0,128,0,0.3)', width=1),
                                    showlegend=False
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=peak_hours['timestamp'],
                                    y=peak_hours['mean'] - peak_hours['std'],
                                    mode='lines',
                                    name='Lower Bound',
                                    line=dict(color='rgba(0,128,0,0.3)', width=1),
                                    fill='tonexty',
                                    fillcolor='rgba(0,128,0,0.1)',
                                    showlegend=False
                                ))
                            
                            fig.update_layout(
                                title="Peak Hour AQI Analysis with Standard Deviation",
                                xaxis_title="Hour of Day",
                                yaxis_title="AQI Value",
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient data for peak hour analysis")
                    else:
                        st.info("Peak hour analysis not available")
                except Exception as e:
                    st.error(f"Error in peak hour analysis: {e}")
            
            # Advanced pollutant insights
            st.subheader("🌫️ Advanced Pollutant Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Pollutant concentration ranges
                    if 'pm2_5' in df.columns and 'pm10' in df.columns:
                        # Filter out invalid values
                        valid_pm25 = df[df['pm2_5'] > 0]['pm2_5']
                        
                        if len(valid_pm25) > 0:
                            # Create concentration categories
                            df['pm25_category'] = pd.cut(valid_pm25, 
                                                       bins=[0, 12, 35.4, 55.4, 150.4, 250.4, 500],
                                                       labels=['Good', 'Moderate', 'Unhealthy SG', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
                            
                            pm25_dist = df['pm25_category'].value_counts()
                            
                            if len(pm25_dist) > 0:
                                fig = go.Figure(data=go.Pie(
                                    labels=pm25_dist.index,
                                    values=pm25_dist.values,
                                    hole=0.3,
                                    marker_colors=['green', 'yellow', 'orange', 'red', 'purple', 'maroon']
                                ))
                                
                                fig.update_layout(
                                    title="PM2.5 Concentration Categories Distribution",
                                    height=400,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No PM2.5 categories available")
                        else:
                            st.info("No valid PM2.5 data for category analysis")
                    else:
                        st.info("PM2.5 category analysis not available")
                except Exception as e:
                    st.error(f"Error in PM2.5 category analysis: {e}")
            
                try:
                    # Pollutant exceedance analysis
                    if 'pm2_5' in df.columns and 'pm10' in df.columns:
                        # Filter out invalid values
                        valid_pm25 = df[df['pm2_5'] > 0]
                        valid_pm10 = df[df['pm10'] > 0]
                        
                        if len(valid_pm25) > 0 and len(valid_pm10) > 0:
                            # WHO guidelines: PM2.5 < 10 μg/m³, PM10 < 20 μg/m³
                            who_pm25_exceedance = (valid_pm25['pm2_5'] > 10).sum() / len(valid_pm25) * 100
                            who_pm10_exceedance = (valid_pm10['pm10'] > 20).sum() / len(valid_pm10) * 100
                            
                            # EPA standards: PM2.5 < 12 μg/m³, PM10 < 54 μg/m³
                            epa_pm25_exceedance = (valid_pm25['pm2_5'] > 12).sum() / len(valid_pm25) * 100
                            epa_pm10_exceedance = (valid_pm10['pm10'] > 54).sum() / len(valid_pm10) * 100
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=['WHO PM2.5', 'WHO PM10', 'EPA PM2.5', 'EPA PM10'],
                                y=[who_pm25_exceedance, who_pm10_exceedance, epa_pm25_exceedance, epa_pm10_exceedance],
                                marker_color=['red', 'red', 'orange', 'orange'],
                                name='Exceedance Percentage'
                            ))
                            
                            fig.update_layout(
                                title="Pollutant Exceedance Analysis (% of time above standards)",
                                yaxis_title="Exceedance Percentage (%)",
                                height=400,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show exceedance statistics
                            st.write("**Exceedance Statistics:**")
                            st.write(f"• WHO PM2.5 (>10 μg/m³): {who_pm25_exceedance:.1f}%")
                            st.write(f"• WHO PM10 (>20 μg/m³): {who_pm10_exceedance:.1f}%")
                            st.write(f"• EPA PM2.5 (>12 μg/m³): {epa_pm25_exceedance:.1f}%")
                            st.write(f"• EPA PM10 (>54 μg/m³): {epa_pm10_exceedance:.1f}%")
                        else:
                            st.info("Insufficient valid pollutant data for exceedance analysis")
                    else:
                        st.info("Pollutant exceedance analysis not available")
                except Exception as e:
                    st.error(f"Error in pollutant exceedance analysis: {e}")
                    
        except Exception as e:
            st.error(f"Error in additional insights: {e}")
            st.info("Please check the console for detailed error information.")
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        try:
            # Basic statistics for numeric columns
            numeric_summary = df.describe()
            st.write("**Numeric Columns Summary:**")
            
            # Convert to string format to avoid PyArrow issues
            try:
                st.dataframe(numeric_summary, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying statistics table: {e}")
                # Fallback: display as formatted text
                st.write("**Summary Statistics:**")
                for col in numeric_summary.columns:
                    st.write(f"**{col}:**")
                    st.write(f"  - Count: {numeric_summary[col]['count']:.0f}")
                    st.write(f"  - Mean: {numeric_summary[col]['mean']:.2f}")
                    st.write(f"  - Std: {numeric_summary[col]['std']:.2f}")
                    st.write(f"  - Min: {numeric_summary[col]['min']:.2f}")
                    st.write(f"  - 25%: {numeric_summary[col]['25%']:.2f}")
                    st.write(f"  - 50%: {numeric_summary[col]['50%']:.2f}")
                    st.write(f"  - 75%: {numeric_summary[col]['75%']:.2f}")
                    st.write(f"  - Max: {numeric_summary[col]['max']:.2f}")
                    st.write("---")
                
        except Exception as e:
            st.error(f"Error in statistical summary: {e}")
        
        # Download section
        st.subheader("💾 Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Processed Data"):
                try:
                    # Create a processed version of the data
                    processed_df = df.copy()
                    
                    # Convert to CSV for download
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name="processed_historical_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating download: {e}")
        
        with col2:
            if st.button("📊 Download Summary Report"):
                try:
                    # Create a summary report
                    report = f"""
Historical Data Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
- Total Records: {len(df)}
- Date Range: {str(df['timestamp'].min())[:10]} to {str(df['timestamp'].max())[:10]}
- Columns: {len(df.columns)}
- Missing Values: {df.isnull().sum().sum()}

Key Statistics:
- Average AQI: {df['aqi'].mean():.2f}
- AQI Range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}
- Data Completeness: {(df.notnull().sum() / len(df) * 100).mean():.1f}%

Analysis completed successfully.
                    """
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name="historical_data_report.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error creating report: {e}")
                    
    except Exception as e:
        st.error(f"❌ Critical error in Historical EDA: {e}")
        st.info("Please check the console for detailed error information.")
        st.exception(e)

def show_creator_info():
    """Show information about the project creator"""
    st.header("👨‍💻 Project Creator")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="text-align: center; margin-bottom: 1.5rem;">🚀 AQI Forecasting System</h2>
        <div style="text-align: center;">
            <h3 style="margin-bottom: 1rem;">Created by: <strong>Muhammad Adeel</strong></h3>
            <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">
                A comprehensive air quality forecasting system using machine learning and real-time data collection.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Creator details in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 1.2rem;">👨‍💻 Developer</h3>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">Muhammad Adeel</p>
            <p style="margin: 0; font-size: 0.9rem;">Full Stack Developer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: #333; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 1.2rem;">🔗 LinkedIn</h3>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">muhammadadeel21</p>
            <p style="margin: 0; font-size: 0.9rem;">Professional Profile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; color: #333; margin: 0.5rem 0;">
            <h3 style="margin: 0; font-size: 1.2rem;">🐙 GitHub</h3>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">adeelkh21</p>
            <p style="margin: 0; font-size: 0.9rem;">Open Source Projects</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Project description
    st.subheader("📋 About This Project")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h3 style="margin: 0 0 1rem 0;">🌤️ Air Quality Index Forecasting System</h3>
        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
            <strong>Purpose:</strong> Real-time air quality monitoring and 72-hour forecasting for Peshawar, Pakistan
        </p>
        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
            <strong>Technology:</strong> Machine Learning (Random Forest, Gradient Boosting, LSTM), FastAPI, Streamlit
        </p>
        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
            <strong>Data:</strong> Real-time weather and pollution data collection with historical analysis
        </p>
        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
            <strong>Features:</strong> Live dashboard, automated forecasting, comprehensive EDA, and real-time updates
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information
    st.subheader("📞 Contact Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); 
                    padding: 1rem; border-radius: 10px; color: white;">
            <h4 style="margin: 0 0 1rem 0;">🔗 Connect with Muhammad Adeel</h4>
            <p style="margin: 0.5rem 0;">• <strong>LinkedIn:</strong> muhammadadeel21</p>
            <p style="margin: 0.5rem 0;">• <strong>GitHub:</strong> adeelkh21</p>
            <p style="margin: 0.5rem 0;">• <strong>Project:</strong> AQI Forecasting System</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 10px; color: white;">
            <h4 style="margin: 0 0 1rem 0;">💡 Project Highlights</h4>
            <p style="margin: 0.5rem 0;">• <strong>Real-time Data:</strong> Live weather & pollution collection</p>
            <p style="margin: 0.5rem 0;">• <strong>ML Models:</strong> Ensemble forecasting approach</p>
            <p style="margin: 0.5rem 0;">• <strong>Web Interface:</strong> Interactive Streamlit dashboard</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application with real-time functionality"""
    # Display header with author information
    show_header()
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🌤️ Navigation</h2>
        <p style="color: #667eea; font-weight: bold;">Real-Time AQI System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page selection
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Analytics", "Historical EDA", "Creator Info"],
        index=0 if st.session_state.page == "Dashboard" else 1 if st.session_state.page == "Analytics" else 2 if st.session_state.page == "Historical EDA" else 3
    )
    
    st.session_state.page = page
    
    # Page descriptions
    if page == "Dashboard":
        st.sidebar.info("📊 **Dashboard**: Real-time AQI monitoring and forecasting")
    elif page == "Analytics":
        st.sidebar.info("📈 **Analytics**: Forecast insights and model performance")
    elif page == "Historical EDA":
        st.sidebar.info("🔍 **Historical EDA**: Deep analysis of historical data patterns")
    elif page == "Creator Info":
        st.sidebar.info("👨‍💻 **Creator Info**: About the developer and project")
    
    # Display selected page
    if page == "Dashboard":
        st.markdown("## 📊 Real-Time Dashboard")
        show_real_time_dashboard()
    elif page == "Analytics":
        st.markdown("## 📈 Analytics & Insights")
        show_analytics()
    elif page == "Historical EDA":
        st.markdown("## 🔍 Historical Data Analysis")
        show_historical_eda()
    elif page == "Creator Info":
        st.markdown("## 👨‍💻 About the Creator")
        show_creator_info()
    
    # Enhanced footer with real-time info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status:**")
    
    if check_api_health():
        st.sidebar.success("✅ Backend: Online")
    else:
        st.sidebar.error("❌ Backend: Offline")
    
    st.sidebar.markdown(f"**Last Check:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-mode status
    if st.session_state.auto_mode:
        st.sidebar.success("🤖 Auto-Mode: Enabled")
    else:
        st.sidebar.warning("⏸️ Auto-Mode: Disabled")
    
    # Last update times
    if st.session_state.last_data_collection:
        st.sidebar.markdown(f"**Last Data:** {st.session_state.last_data_collection.strftime('%H:%M:%S')}")
    
    if st.session_state.last_forecast:
        st.sidebar.markdown(f"**Last Forecast:** {st.session_state.last_forecast.strftime('%H:%M:%S')}")
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Info:**")
    st.sidebar.markdown("🌍 Location: Peshawar")
    st.sidebar.markdown("🤖 Models: 3 Active")
    st.sidebar.markdown("📊 Data: Real-time")
    st.sidebar.markdown("🔮 Forecast: 72 hours")
    st.sidebar.markdown("🔄 Auto-refresh: 2 min")

if __name__ == "__main__":
    main()
