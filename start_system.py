"""
Startup Script for AQI Forecasting System
========================================

This script can launch both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI Backend...")
    try:
        # Change to the api directory
        api_dir = Path(__file__).parent / "api"
        os.chdir(api_dir)
        
        # Start the backend
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ Backend started with PID: {process.pid}")
        print("🌐 API available at: http://localhost:8001")
        print("📚 API docs at: http://localhost:8001/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit Frontend...")
    try:
        # Change back to the root directory
        root_dir = Path(__file__).parent
        os.chdir(root_dir)
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ Frontend started with PID: {process.pid}")
        print("🌐 Dashboard available at: http://localhost:8501")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def wait_for_backend():
    """Wait for backend to be ready"""
    import requests
    max_attempts = 30
    attempt = 0
    
    print("⏳ Waiting for backend to be ready...")
    
    while attempt < max_attempts:
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except:
            pass
        
        attempt += 1
        time.sleep(2)
        print(f"   Attempt {attempt}/{max_attempts}...")
    
    print("❌ Backend failed to start within expected time")
    return False

def main():
    """Main startup function"""
    print("🌤️ AQI Forecasting System Startup")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        "api/main.py",
        "streamlit_app.py",
        "phase1_data_collection.py",
        "enhanced_aqi_forecasting_real.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure all files are present before starting the system.")
        return
    
    print("✅ All required files found")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Cannot start frontend without backend")
        return
    
    # Wait for backend to be ready
    if not wait_for_backend():
        print("❌ Backend failed to start properly")
        backend_process.terminate()
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        return
    
    print("\n🎉 System started successfully!")
    print("=" * 40)
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
        print("\n🛑 Shutting down system...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
            
        print("👋 System shutdown complete")

if __name__ == "__main__":
    main()
