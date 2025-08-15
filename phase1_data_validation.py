"""
Data Validation Module for AQI Prediction System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger('validation')

class DataValidator:
    def __init__(self):
        """Initialize data validator with expected schemas and value ranges"""
        # Expected columns and their types
        self.required_columns = {
            'timestamp': 'datetime64[ns]',
            'aqi_category': 'int64',
            'pm2_5': 'float64',
            'pm10': 'float64',
            'temperature': 'float64',
            'relative_humidity': 'float64',
            'wind_speed': 'float64',
            'pressure': 'float64'
        }
        
        # Value ranges for validation
        self.value_ranges = {
            'temperature': (-30, 60),  # °C
            'relative_humidity': (0, 100),  # %
            'wind_speed': (0, 100),  # m/s
            'pressure': (800, 1200),  # hPa
            'pm2_5': (0, 1000),  # μg/m³
            'pm10': (0, 1000),  # μg/m³
            'aqi_category': (1, 5)  # AQI categories
        }

    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate merged dataset
        
        Args:
            df (pd.DataFrame): Merged DataFrame to validate
            
        Returns:
            Tuple[bool, Dict]: (is_valid, validation report)
        """
        report = {
            'completeness': {},
            'consistency': {},
            'quality_metrics': {}
        }
        
        try:
            # Check data completeness
            total_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            expected_records = int(total_hours) + 1
            actual_records = len(df)
            
            report['completeness'] = {
                'expected_records': expected_records,
                'actual_records': actual_records,
                'coverage_ratio': actual_records / expected_records if expected_records > 0 else 0
            }
            
            # Check data consistency
            report['consistency'] = {
                'timestamp_gaps': self._check_timestamp_gaps(df),
                'value_ranges': self._check_value_ranges(df)
            }
            
            # Calculate quality metrics
            report['quality_metrics'] = {
                'missing_rate': df.isnull().mean().to_dict(),
                'unique_values': df.nunique().to_dict()
            }
            
            # Calculate AQI values if not present
            if 'aqi_numeric' not in df.columns:
                df = self._calculate_aqi(df)
            
            # Define optional fields that can be missing
            optional_fields = {'snow', 'wpgt', 'tsun', 'coco'}
            
            # Check missing rates excluding optional fields
            missing_rates = {k: v for k, v in report['quality_metrics']['missing_rate'].items() 
                           if k not in optional_fields}
            
            # Determine if data is valid
            is_valid = (
                # Data completeness checks
                report['completeness']['coverage_ratio'] >= 0.90 and  # At least 90% coverage
                report['consistency']['timestamp_gaps']['max_gap_hours'] <= 3 and  # No gaps > 3 hours
                
                # Missing value checks (excluding optional fields)
                all(v <= 0.1 for k, v in missing_rates.items() if k not in optional_fields) and
                
                # Core field checks
                df['pm2_5'].notnull().all() and  # No missing PM2.5
                df['pm10'].notnull().all() and   # No missing PM10
                df['temperature'].notnull().mean() >= 0.9 and  # Max 10% missing temp
                df['relative_humidity'].notnull().mean() >= 0.9 and  # Max 10% missing humidity
                
                # Value range checks
                all(report['consistency']['value_ranges'].values())
            )
            
            if not is_valid:
                logger.warning("Data validation failed")
                logger.warning(f"Validation report: {report}")
            else:
                logger.info("Data validation passed")
                logger.info(f"Data quality metrics: {report['quality_metrics']}")
            
            return is_valid, report
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise

    def _check_timestamp_gaps(self, df: pd.DataFrame) -> Dict:
        """Check for gaps in timestamp sequence"""
        df = df.sort_values('timestamp')
        time_diff = df['timestamp'].diff()
        
        return {
            'max_gap_hours': time_diff.max().total_seconds() / 3600,
            'avg_gap_hours': time_diff.mean().total_seconds() / 3600,
            'gaps_over_1h': (time_diff > pd.Timedelta(hours=1)).sum()
        }

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check if values are within expected ranges"""
        results = {}
        for column, (min_val, max_val) in self.value_ranges.items():
            if column in df.columns:
                valid_values = df[column].between(min_val, max_val)
                results[column] = valid_values.all()
        return results

    def _calculate_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AQI if not present"""
        def calculate_aqi(concentration, breakpoints):
            for _, (low_conc, high_conc, low_aqi, high_aqi) in breakpoints.items():
                if low_conc <= concentration <= high_conc:
                    return np.interp(concentration, [low_conc, high_conc], [low_aqi, high_aqi])
            return 500

        # PM2.5 breakpoints
        pm25_breakpoints = {
            'Good': (0.0, 12.0, 0, 50),
            'Moderate': (12.1, 35.4, 51, 100),
            'Unhealthy for Sensitive Groups': (35.5, 55.4, 101, 150),
            'Unhealthy': (55.5, 150.4, 151, 200),
            'Very Unhealthy': (150.5, 250.4, 201, 300),
            'Hazardous': (250.5, 500.4, 301, 500)
        }
        
        # PM10 breakpoints
        pm10_breakpoints = {
            'Good': (0, 54, 0, 50),
            'Moderate': (55, 154, 51, 100),
            'Unhealthy for Sensitive Groups': (155, 254, 101, 150),
            'Unhealthy': (255, 354, 151, 200),
            'Very Unhealthy': (355, 424, 201, 300),
            'Hazardous': (425, 604, 301, 500)
        }

        df['aqi_pm25'] = df['pm2_5'].apply(lambda x: calculate_aqi(x, pm25_breakpoints))
        df['aqi_pm10'] = df['pm10'].apply(lambda x: calculate_aqi(x, pm10_breakpoints))
        df['aqi_numeric'] = df[['aqi_pm25', 'aqi_pm10']].max(axis=1)
        
        return df