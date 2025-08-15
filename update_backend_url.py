#!/usr/bin/env python3
"""
Script to update backend URL in Streamlit app for production deployment
"""

import re
import os

def update_backend_url(new_url):
    """Update backend URL in streamlit_app_clean.py"""
    
    # Read the current file
    with open('streamlit_app_clean.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace localhost URLs with new backend URL
    old_patterns = [
        r'http://localhost:8001',
        r'http://127.0.0.1:8001',
        r'localhost:8001'
    ]
    
    updated_content = content
    for pattern in old_patterns:
        updated_content = re.sub(pattern, new_url, updated_content)
    
    # Write the updated content back
    with open('streamlit_app_clean.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ Backend URL updated to: {new_url}")
    print("📝 File: streamlit_app_clean.py")
    
    # Show what was changed
    if content != updated_content:
        print("\n🔄 Changes made:")
        for pattern in old_patterns:
            if re.search(pattern, content):
                print(f"   - Replaced {pattern} with {new_url}")
    else:
        print("\nℹ️  No changes needed - backend URL already updated")

def main():
    """Main function to update backend URL"""
    print("🚀 Backend URL Update Script")
    print("=" * 40)
    
    # Get the new backend URL from user
    print("\nEnter your backend URL (e.g., https://your-app.railway.app):")
    new_url = input("Backend URL: ").strip()
    
    if not new_url:
        print("❌ Backend URL cannot be empty")
        return
    
    # Validate URL format
    if not new_url.startswith(('http://', 'https://')):
        print("❌ URL must start with http:// or https://")
        return
    
    # Update the file
    try:
        update_backend_url(new_url)
        
        print(f"\n🎉 Successfully updated backend URL!")
        print(f"📱 Your Streamlit app will now connect to: {new_url}")
        print("\n📋 Next steps:")
        print("1. Commit and push the changes:")
        print("   git add .")
        print("   git commit -m 'Update: Backend URL for production'")
        print("   git push origin main")
        print("2. Streamlit Cloud will automatically redeploy")
        print("3. Test your app - forecasting should now work!")
        
    except Exception as e:
        print(f"❌ Error updating backend URL: {e}")
        print("   Please check if streamlit_app_clean.py exists and is writable")

if __name__ == "__main__":
    main()
