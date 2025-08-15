#!/usr/bin/env python3
"""
Deployment Startup Script for AQI Forecasting System
This script helps you prepare and deploy your project to GitHub and Streamlit Cloud
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def print_banner():
    """Print a beautiful banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 AQI Forecasting System                ║
    ║                     Deployment Helper                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install Git first.")
        print("   Download from: https://git-scm.com/downloads")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def check_essential_files():
    """Check if essential files exist"""
    essential_files = [
        "streamlit_app_clean.py",
        "enhanced_aqi_forecasting_real.py",
        "phase1_data_collection.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All essential files are present")
        return True

def initialize_git_repo():
    """Initialize git repository if not already done"""
    if os.path.exists(".git"):
        print("✅ Git repository already initialized")
        return True
    
    try:
        subprocess.run(["git", "init"], check=True)
        print("✅ Git repository initialized")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to initialize git repository")
        return False

def create_initial_commit():
    """Create initial commit if not already done"""
    try:
        # Check if there are any commits
        result = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Git repository already has commits")
            return True
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        
        # Create initial commit
        subprocess.run(["git", "commit", "-m", "Initial commit: AQI Forecasting System"], check=True)
        print("✅ Initial commit created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create initial commit")
        return False

def get_github_repo_info():
    """Get GitHub repository information from user"""
    print("\n📋 GitHub Repository Setup")
    print("=" * 50)
    
    username = input("Enter your GitHub username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return None
    
    repo_name = input("Enter repository name (default: aqi-forecasting-system): ").strip()
    if not repo_name:
        repo_name = "aqi-forecasting-system"
    
    return {
        "username": username,
        "repo_name": repo_name,
        "url": f"https://github.com/{username}/{repo_name}.git"
    }

def setup_remote_origin(repo_info):
    """Setup remote origin for GitHub"""
    try:
        # Check if remote already exists
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if "origin" in result.stdout:
            print("✅ Remote origin already configured")
            return True
        
        # Add remote origin
        subprocess.run(["git", "remote", "add", "origin", repo_info["url"]], check=True)
        print("✅ Remote origin configured")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to configure remote origin")
        return False

def push_to_github():
    """Push code to GitHub"""
    try:
        # Set main branch
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        
        # Push to GitHub
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("✅ Code pushed to GitHub successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to push to GitHub")
        print("   Make sure you have access to the repository and have set up authentication")
        return False

def show_deployment_instructions(repo_info):
    """Show deployment instructions"""
    print("\n🎉 Deployment Instructions")
    print("=" * 50)
    print(f"✅ Your code is now on GitHub: {repo_info['url']}")
    print("\n🌐 Next Steps for Streamlit Cloud:")
    print("1. Go to https://streamlit.io/cloud")
    print("2. Sign in with GitHub")
    print("3. Click 'New app'")
    print(f"4. Select repository: {repo_info['username']}/{repo_info['repo_name']}")
    print("5. Set main file path: streamlit_app_clean.py")
    print("6. Click 'Deploy!'")
    print("\n🔗 Your app will be available at:")
    print(f"   https://{repo_info['repo_name']}.streamlit.app")
    print("\n📚 For detailed instructions, see DEPLOYMENT.md")

def main():
    """Main deployment process"""
    print_banner()
    
    # Check prerequisites
    print("🔍 Checking Prerequisites...")
    if not check_git_installed():
        return
    if not check_python_version():
        return
    if not check_essential_files():
        return
    
    print("\n✅ All prerequisites met!")
    
    # Git setup
    print("\n🔧 Setting up Git Repository...")
    if not initialize_git_repo():
        return
    if not create_initial_commit():
        return
    
    # GitHub setup
    repo_info = get_github_repo_info()
    if not repo_info:
        return
    
    if not setup_remote_origin(repo_info):
        return
    
    # Push to GitHub
    print(f"\n🚀 Pushing to GitHub: {repo_info['url']}")
    if not push_to_github():
        return
    
    # Show deployment instructions
    show_deployment_instructions(repo_info)
    
    print("\n🎊 Deployment setup complete!")
    print("   Your AQI Forecasting System is ready for Streamlit Cloud!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Deployment setup cancelled by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   Please check the error and try again")
