"""
Feature Store Integration with Hopsworks
========================================

This module handles the integration with Hopsworks Feature Store
for storing and retrieving engineered features for AQI prediction.
"""

import os
import pandas as pd
import hopsworks
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreManager:
    """
    Manages interaction with Hopsworks Feature Store
    """
    
    def __init__(self, project_name: str = "aqi_prediction_peshawar"):
        """
        Initialize connection to Hopsworks Feature Store
        
        Args:
            project_name: Name of the Hopsworks project
        """
        self.project_name = project_name
        self.project = None
        self.fs = None
        
    def connect(self) -> bool:
        """
        Connect to Hopsworks Feature Store
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Get API key from environment
            api_key = os.getenv("HOPSWORKS_API_KEY")
            if not api_key:
                logger.warning("HOPSWORKS_API_KEY not found in environment variables")
                return False
            
            # Login to Hopsworks
            self.project = hopsworks.login(
                project=self.project_name,
                api_key_value=api_key
            )
            
            # Get feature store
            self.fs = self.project.get_feature_store()
            
            logger.info(f"✅ Connected to Hopsworks project: {self.project_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Hopsworks: {e}")
            return False
    
    def create_feature_validation(self) -> Dict:
        """
        Create validation rules for AQI features
        
        Returns:
            Dict: Validation rules dictionary
        """
        validation_rules = {
            "aqi_numeric": {"min": 0, "max": 500},
            "pm2_5": {"min": 0, "max": 1000},
            "pm10": {"min": 0, "max": 2000},
            "no2": {"min": 0, "max": 500},
            "o3": {"min": 0, "max": 500},
            "temperature": {"min": -50, "max": 60},
            "relative_humidity": {"min": 0, "max": 100},
            "wind_speed": {"min": 0, "max": 100},
            "pressure": {"min": 800, "max": 1200}
        }
        
        return validation_rules
    
    def prepare_feature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for feature store upload
        
        Args:
            df: DataFrame with features
            
        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        df = df.copy()
        
        # Add ID column as primary key
        df['id'] = range(len(df))
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def create_feature_groups(self, df: pd.DataFrame) -> bool:
        """
        Create feature groups in Hopsworks
        
        Args:
            df: DataFrame with features
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.fs:
                logger.error("Not connected to feature store")
                return False
            
            # Prepare data
            df_prepared = self.prepare_feature_data(df)
            
            # Create AQI features group
            aqi_fg = self.fs.get_or_create_feature_group(
                name="aqi_features",
                version=1,
                description="AQI prediction features including weather and pollution data",
                primary_key=["id"],
                event_time="timestamp",
                statistics_config=True
            )
            
            # Insert data
            aqi_fg.insert(df_prepared, write_options={"wait_for_job": False})
            
            logger.info(f"✅ Created feature group with {len(df_prepared)} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature groups: {e}")
            return False
    
    def get_feature_group(self, name: str, version: int = 1):
        """
        Get existing feature group
        
        Args:
            name: Feature group name
            version: Feature group version
            
        Returns:
            Feature group object or None
        """
        try:
            if not self.fs:
                logger.error("Not connected to feature store")
                return None
            
            fg = self.fs.get_feature_group(name=name, version=version)
            logger.info(f"✅ Retrieved feature group: {name} v{version}")
            return fg
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature group {name}: {e}")
            return None
    
    def read_features(self, 
                     feature_group_name: str = "aqi_features",
                     version: int = 1) -> Optional[pd.DataFrame]:
        """
        Read features from feature store
        
        Args:
            feature_group_name: Name of feature group
            version: Version of feature group
            
        Returns:
            pd.DataFrame: Features data or None
        """
        try:
            fg = self.get_feature_group(feature_group_name, version)
            if fg is None:
                return None
            
            # Read data
            df = fg.read()
            logger.info(f"✅ Read {len(df)} records from feature store")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to read features: {e}")
            return None
    
    def upload_features(self, 
                       features_path: str,
                       feature_group_name: str = "aqi_features") -> bool:
        """
        Upload features from CSV file to feature store
        
        Args:
            features_path: Path to features CSV file
            feature_group_name: Name for feature group
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load features
            df = pd.read_csv(features_path)
            logger.info(f"📊 Loaded {len(df)} records from {features_path}")
            
            # Connect if not already connected
            if not self.fs:
                if not self.connect():
                    return False
            
            # Create feature groups
            success = self.create_feature_groups(df)
            
            if success:
                logger.info("🎉 Features successfully uploaded to Hopsworks!")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to upload features: {e}")
            return False

def main():
    """
    Main function for testing feature store operations
    """
    # Initialize feature store manager
    fs_manager = FeatureStoreManager()
    
    # Connect to Hopsworks
    if not fs_manager.connect():
        logger.error("Failed to connect to Hopsworks")
        return
    
    # Example: Upload features (uncomment when ready)
    # features_path = "data_repositories/features/simple_features.csv"
    # if os.path.exists(features_path):
    #     fs_manager.upload_features(features_path)
    # else:
    #     logger.warning(f"Features file not found: {features_path}")

if __name__ == "__main__":
    main()
