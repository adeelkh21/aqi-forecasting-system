#!/usr/bin/env python3
"""
Debug script to test alternative Meteostat approaches
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from meteostat import Point, Hourly

# Peshawar coordinates
PESHAWAR_LAT = 34.0083
PESHAWAR_LON = 71.5189

def test_meteostat_alternatives():
    """Test different Meteostat approaches to avoid flag columns issue"""
    print("🔍 Testing Alternative Meteostat Approaches")
    print("=" * 60)
    
    # Test 1: Try with different time periods
    print("\n🧪 Test 1: Different Time Periods")
    print("-" * 40)
    
    test_periods = [
        (1, "1 day"),
        (7, "1 week"), 
        (30, "1 month")
    ]
    
    for days, description in test_periods:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"\n📅 Testing {description}: {start_date.date()} to {end_date.date()}")
            
            location = Point(PESHAWAR_LAT, PESHAWAR_LON)
            data = Hourly(location, start_date, end_date)
            df = data.fetch()
            
            if df is not None and not df.empty:
                print(f"✅ Success! Shape: {df.shape}, Columns: {list(df.columns)}")
                break
            else:
                print("❌ No data received")
                
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}...")
    
    # Test 2: Try with different location (nearby city)
    print("\n🧪 Test 2: Different Location")
    print("-" * 40)
    
    # Islamabad coordinates (nearby)
    ISLAMABAD_LAT = 33.7294
    ISLAMABAD_LON = 73.0931
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        print(f"📍 Testing Islamabad ({ISLAMABAD_LAT}, {ISLAMABAD_LON})")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        
        location = Point(ISLAMABAD_LAT, ISLAMABAD_LON)
        data = Hourly(location, start_date, end_date)
        df = data.fetch()
        
        if df is not None and not df.empty:
            print(f"✅ Success! Shape: {df.shape}, Columns: {list(df.columns)}")
            print("📊 Sample columns:", list(df.columns)[:5])
        else:
            print("❌ No data received")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
    
    # Test 3: Try with Daily instead of Hourly
    print("\n🧪 Test 3: Daily Data Instead of Hourly")
    print("-" * 40)
    
    try:
        from meteostat import Daily
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        
        location = Point(PESHAWAR_LAT, PESHAWAR_LON)
        data = Daily(location, start_date, end_date)
        df = data.fetch()
        
        if df is not None and not df.empty:
            print(f"✅ Success! Shape: {df.shape}, Columns: {list(df.columns)}")
            print("📊 Sample columns:", list(df.columns)[:5])
        else:
            print("❌ No data received")
            
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")

if __name__ == "__main__":
    test_meteostat_alternatives()
