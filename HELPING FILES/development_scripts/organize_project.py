"""
Project Organization Script - Clean up and organize files
========================================================

This script moves helper/intermediate files to "HELPING FILES" folder
and keeps only the essential files that achieved our Phase 2 goals.
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """Organize project files into essential and helper categories"""
    
    print("🗂️  ORGANIZING PROJECT FILES")
    print("=" * 35)
    
    # Define core/essential files (keep in root)
    essential_files = {
        # Core pipeline files (final versions)
        'collect_historical_data.py',    # Historical data collection
        'data_collection.py',            # Hourly data collection  
        'merge_data.py',                 # Data merging pipeline
        'final_feature_engineering.py', # Final feature engineering (our achievement)
        'data_validation.py',            # Data validation
        'logging_config.py',             # Logging configuration
        
        # Configuration and documentation
        'requirements.txt',              # Dependencies
        'roadmap.md',                   # Project roadmap
        'notes.md',                     # Project notes
        
        # Core data files (in data_repositories)
        # - final_features.csv (our main achievement)
        # - final_performance.json (validation results)
        # - clean_features.csv (validated dataset)
    }
    
    # Define helper/intermediate files (move to HELPING FILES)
    helper_files = {
        # Development and testing files
        'feature_engineering.py',        # Original feature engineering
        'enhanced_feature_engineering.py', # Intermediate version
        'evaluate_features.py',          # Performance evaluation
        'test_enhanced_features.py',     # Testing script
        'validate_model_integrity.py',   # Validation script
        'fix_feature_leakage.py',        # Leakage fixing script
        
        # Other development files
        'api_utils.py',                  # API utilities
        'feature_store.py',              # Feature store (Phase 3)
        'model_training.py',             # Model training (Phase 4)
        'prediction_service.py',         # Prediction service (Phase 5)
        'REVISED_ROADMAP.md',            # Revised roadmap
        
        # Temporary organization script
        'organize_project.py'            # This script itself
    }
    
    # Create helper directories
    helping_dir = Path("HELPING FILES")
    helping_dir.mkdir(exist_ok=True)
    
    # Create subdirectories in HELPING FILES
    (helping_dir / "development_scripts").mkdir(exist_ok=True)
    (helping_dir / "testing_scripts").mkdir(exist_ok=True)
    (helping_dir / "intermediate_features").mkdir(exist_ok=True)
    (helping_dir / "analysis_scripts").mkdir(exist_ok=True)
    
    print("\n📁 Moving Helper Files:")
    print("-" * 25)
    
    # Move helper files
    moved_files = 0
    for file in helper_files:
        if os.path.exists(file):
            # Categorize files
            if 'test_' in file or 'evaluate' in file or 'validate' in file:
                dest_dir = helping_dir / "testing_scripts"
            elif 'enhanced' in file or 'fix_' in file:
                dest_dir = helping_dir / "analysis_scripts"  
            else:
                dest_dir = helping_dir / "development_scripts"
            
            dest_path = dest_dir / file
            shutil.move(file, str(dest_path))
            print(f"   ✅ {file} → {dest_path}")
            moved_files += 1
    
    # Handle data_repositories organization
    print(f"\n📊 Organizing Data Repositories:")
    print("-" * 35)
    
    data_repo = Path("data_repositories")
    if data_repo.exists():
        helping_data = helping_dir / "intermediate_data"
        helping_data.mkdir(exist_ok=True)
        
        # Move intermediate feature files
        features_dir = data_repo / "features"
        if features_dir.exists():
            intermediate_features = [
                'engineered_features.csv',    # Original engineered features
                'enhanced_features.csv',      # Enhanced features (with leakage)
                'clean_features_metadata.json', # Metadata for clean features
                'feature_importance.csv',     # Feature importance analysis
                'feature_metadata.json'      # Original metadata
            ]
            
            for feat_file in intermediate_features:
                feat_path = features_dir / feat_file
                if feat_path.exists():
                    dest_path = helping_data / feat_file
                    shutil.move(str(feat_path), str(dest_path))
                    print(f"   ✅ {feat_file} → intermediate_data/")
                    moved_files += 1
        
        # Move feature analysis directory
        analysis_dir = data_repo / "feature_analysis"
        if analysis_dir.exists():
            dest_analysis = helping_data / "feature_analysis"
            shutil.move(str(analysis_dir), str(dest_analysis))
            print(f"   ✅ feature_analysis/ → intermediate_data/")
            moved_files += 1
    
    # Create final project summary
    print(f"\n📋 Essential Files Kept in Root:")
    print("-" * 35)
    
    for file in sorted(essential_files):
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    print(f"\n📂 Final Data Structure:")
    print("-" * 25)
    print(f"   data_repositories/")
    print(f"   ├── historical_data/        # Raw historical data")
    print(f"   ├── merged_data/            # Merged datasets")
    print(f"   └── features/")
    print(f"       ├── final_features.csv    # 🎯 MAIN ACHIEVEMENT")
    print(f"       ├── clean_features.csv    # ✅ Validated dataset")
    print(f"       └── final_performance.json # 📊 Results")
    
    print(f"\n✅ Organization Complete!")
    print(f"   Files moved: {moved_files}")
    print(f"   Essential files kept: {len([f for f in essential_files if os.path.exists(f)])}")
    print(f"   Helper files in: HELPING FILES/")
    
    # Create README for HELPING FILES
    readme_content = """# HELPING FILES

This folder contains all intermediate and development files created during Phase 2 feature engineering.

## Directory Structure:

### development_scripts/
- Core development files and utilities
- Feature store and model training preparation files

### testing_scripts/  
- Validation and testing scripts
- Performance evaluation tools
- Model integrity checks

### analysis_scripts/
- Enhanced feature engineering attempts  
- Data leakage fixing scripts
- Feature analysis tools

### intermediate_data/
- Intermediate feature datasets
- Analysis results and metadata
- Feature importance calculations

## Main Achievement Files (in root):
- `final_feature_engineering.py` - Final feature engineering pipeline
- `data_repositories/features/final_features.csv` - Main dataset (215 features, 69.6% R²)
- `data_repositories/features/final_performance.json` - Validation results

## Phase 2 Results:
- ✅ Data leakage issues identified and fixed
- ✅ Realistic validation methodology implemented  
- ✅ 69.6% R² achieved (legitimate performance)
- ✅ 3-day forecasting capability enabled
- ✅ 215 clean, validated features created
"""
    
    with open(helping_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"   📄 Created: HELPING FILES/README.md")

if __name__ == "__main__":
    organize_project()
