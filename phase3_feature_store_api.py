"""
AQI Peshawar - Production Feature Store Integration
=================================================

This module provides production-ready integration with Hopsworks Feature Store
for the AQI prediction system in Peshawar.

Based on the sample feature store code provided.
"""

import os
import pandas as pd
import hopsworks
import hsfs
from datetime import datetime
from typing import Dict, List, Optional

class AQIPeshawarFeatureStore:
    """Production-ready feature store integration for AQI Peshawar"""
    
    def __init__(self, project_name: str = "aqi_prediction_pekhawar"):
        self.project_name = project_name
        self.project = None
        self.fs = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Hopsworks"""
        api_key = os.getenv('HOPSWORKS_API_KEY')
        if not api_key:
            raise ValueError("HOPSWORKS_API_KEY environment variable required")
        
        self.project = hopsworks.login(
            project=self.project_name,
            api_key_value=api_key
        )
        self.fs = self.project.get_feature_store()
    
    def get_feature_group(self, category: str, version: int = 1):
        """Get feature group by category name and version"""
        fg_name = f"aqi_{category}"
        return self.fs.get_feature_group(fg_name, version)
    
    def get_latest_features(self, categories: List[str] = None) -> pd.DataFrame:
        """Get latest features from specified categories or all"""
        if categories is None:
            categories = ['weather', 'pollution', 'temporal', 'lag_features', 'rolling_stats', 'advanced_features']
        
        query_parts = []
        
        for category in categories:
            try:
                fg = self.get_feature_group(category)
                query_parts.append(fg.select_all())
            except Exception as e:
                print(f"Warning: Could not load {category} feature group: {e}")
                continue
        
        if not query_parts:
            raise ValueError("No feature groups available")
        
        # Join all feature groups on timestamp
        final_query = query_parts[0]
        for query in query_parts[1:]:
            final_query = final_query.join(query, on=['timestamp'])
        
        return final_query.read()
    
    def update_features(self, category: str, new_data: pd.DataFrame):
        """Update feature group with new data"""
        fg = self.get_feature_group(category)
        fg.insert(new_data, wait=True)
        return True
    
    def create_training_dataset(self, categories: List[str] = None, 
                               start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Create training dataset from feature groups"""
        features_df = self.get_latest_features(categories)
        
        if start_date:
            features_df = features_df[features_df['timestamp'] >= start_date]
        if end_date:
            features_df = features_df[features_df['timestamp'] <= end_date]
        
        return features_df
    
    def create_feature_view(self, name: str = "aqi_training_view", 
                           categories: List[str] = None, version: int = 1) -> hsfs.feature_view.FeatureView:
        """Create feature view for model training"""
        if categories is None:
            categories = ['weather', 'pollution', 'temporal', 'lag_features']
        
        query_parts = []
        
        for category in categories:
            fg = self.get_feature_group(category)
            query_parts.append(fg.select_all())
        
        # Join all feature groups
        final_query = query_parts[0]
        for query in query_parts[1:]:
            final_query = final_query.join(query, on=['timestamp'])
        
        # Create feature view
        fv = self.fs.create_feature_view(
            name=name,
            version=version,
            query=final_query,
            labels=['aqi_numeric'] if 'aqi_numeric' in final_query.features else []
        )
        
        return fv
    
    def get_feature_statistics(self, category: str) -> Dict:
        """Get statistics for a feature group"""
        fg = self.get_feature_group(category)
        stats = fg.statistics()
        return stats.to_dict() if stats else {}

# Usage examples:
# 
# # Initialize feature store
# fs = AQIPeshawarFeatureStore()
# 
# # Get latest features for prediction
# features = fs.get_latest_features(['weather', 'pollution', 'temporal'])
# 
# # Create training dataset
# training_data = fs.create_training_dataset(
#     categories=['weather', 'pollution', 'lag_features'],
#     start_date='2025-01-01',
#     end_date='2025-08-11'
# )
# 
# # Create feature view for model training
# fv = fs.create_feature_view('aqi_peshawar_training', ['weather', 'pollution', 'temporal'])
# 
# # Update features with new data
# new_weather_data = pd.DataFrame({...})
# fs.update_features('weather', new_weather_data)
