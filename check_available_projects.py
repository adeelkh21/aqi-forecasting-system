"""
Check Available Hopsworks Projects
==================================

This script connects to Hopsworks and lists available projects
to help identify the correct project name.
"""

import os
import hopsworks

def check_projects():
    print("🔍 CHECKING AVAILABLE HOPSWORKS PROJECTS")
    print("=" * 45)
    
    api_key = os.getenv('HOPSWORKS_API_KEY')
    if not api_key:
        print("❌ HOPSWORKS_API_KEY not found in environment")
        return
    
    try:
        print("🔄 Connecting to Hopsworks...")
        
        # Connect without specifying project to list available projects
        connection = hopsworks.login(api_key_value=api_key)
        
        print("✅ Connected successfully!")
        print("\n📋 Available Projects:")
        
        # Get available projects
        projects = connection.get_projects()
        
        if projects:
            for i, project in enumerate(projects, 1):
                print(f"   {i}. {project.name}")
                print(f"      ID: {project.id}")
                print(f"      Created: {project.created}")
                print()
        else:
            print("   No projects found.")
            print("\n💡 You may need to create a new project:")
            print("   1. Go to https://app.hopsworks.ai/")
            print("   2. Click 'Create Project'")
            print("   3. Enter project name (e.g., 'aqi_prediction_peshawar')")
            print("   4. Configure settings and create")
        
        print("\n" + "="*50)
        
        if projects:
            # Try to connect to the first project as an example
            first_project = projects[0]
            print(f"🔄 Testing connection to first project: {first_project.name}")
            
            project_connection = hopsworks.login(
                project=first_project.name,
                api_key_value=api_key
            )
            
            fs = project_connection.get_feature_store()
            print(f"✅ Successfully connected to project: {first_project.name}")
            print(f"🏪 Feature store name: {fs.name}")
            
            return first_project.name
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    project_name = check_projects()
    if project_name:
        print(f"\n💡 Use this project name: {project_name}")
    else:
        print("\n💡 Create a new project in Hopsworks first")
