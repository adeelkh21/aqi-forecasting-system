
"""
Feature Store Pipeline Integration
=================================

This module integrates the feature store with the existing data collection pipeline.
"""

import pandas as pd
from datetime import datetime

class FeatureStorePipeline:
    def __init__(self, feature_store_manager):
        self.fs_manager = feature_store_manager
        
    def update_features_hourly(self, new_data):
        """Update feature store with new hourly data"""
        # Process new data through feature engineering
        engineered_features = self.engineer_features(new_data)
        
        # Update each feature group
        for category, fg_info in self.fs_manager.feature_groups.items():
            category_features = self.extract_category_features(
                engineered_features, category
            )
            # Insert into feature store
            # fg.insert(category_features)
            
    def engineer_features(self, raw_data):
        """Apply feature engineering to new data"""
        # Use existing feature engineering logic
        from final_feature_engineering import create_advanced_features
        return create_advanced_features(raw_data)
        
    def extract_category_features(self, data, category):
        """Extract features for specific category"""
        category_columns = ['timestamp'] + self.fs_manager.feature_categories[category]['features']
        return data[category_columns]
