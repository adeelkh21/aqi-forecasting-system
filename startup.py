#!/usr/bin/env python3
"""
Startup script for AQI Forecasting System
=========================================

This script starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI Backend...")
    
    try:
        # Start backend in background
        process = subprocess.Popen(
            [sys.executable, "api/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Backend started with PID: {process.pid}")
                print("🌐 API available at: http://localhost:8001")
                print("📚 API docs at: http://localhost:8001/docs")
                return process
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                process.terminate()
                return None
        except requests.exceptions.RequestException:
            print("❌ Backend health check failed: Connection refused")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit Frontend...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(5)
        
        print(f"✅ Frontend started with PID: {process.pid}")
        print("📱 Access your dashboard at: http://localhost:8501")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🌤️ AQI Forecasting System Startup")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "api/main.py",
        "streamlit_app.py",
        "requirements.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("✅ All required files found")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Exiting.")
        backend_process.terminate()
        return
    
    print("\n🎉 System startup complete!")
    print("=" * 50)
    print("📱 Access your dashboard at: http://localhost:8501")
    print("🔧 API documentation at: http://localhost:8001/docs")
    print("\n💡 To stop the system, press Ctrl+C")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
