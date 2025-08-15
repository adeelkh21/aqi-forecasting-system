"""
API Utilities for AQI Prediction System
"""

import time
import logging
import requests
from typing import Optional, Dict, Any
from functools import wraps
from requests.exceptions import RequestException

logger = logging.getLogger('api_utils')

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Retry decorator with exponential backoff
    
    Args:
        retries (int): Number of times to retry the wrapped function
        backoff_in_seconds (int): Initial backoff time in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            attempt = 0
            wait_time = backoff_in_seconds
            
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    attempt += 1
                    if attempt == retries:
                        logger.error(f"Failed after {retries} attempts: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                    
        return wrapper
    return decorator

class APIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API client
        
        Args:
            base_url (str): Base URL for the API
            api_key (str, optional): API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    @retry_with_backoff(retries=3)
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Make GET request to API with retry mechanism
        
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            
        Returns:
            dict: API response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            raise
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Error connecting to the server: {str(e)}")
            raise
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out: {str(e)}")
            raise
            
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while making the request: {str(e)}")
            raise
            
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise
