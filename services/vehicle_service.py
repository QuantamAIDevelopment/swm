"""Vehicle service for fetching live vehicle data from SWM API."""
import os
import requests
import pandas as pd
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VehicleService:
    def __init__(self):
        self.base_url = os.getenv('SWM_API_BASE_URL', 'https://uat-swm-main-service-hdaqcdcscbfedhhn.centralindia-01.azurewebsites.net')
        self.api_key = os.getenv('SWM_API_KEY', '')
        self.username = os.getenv('SWM_USERNAME', '')
        self.password = os.getenv('SWM_PASSWORD', '')
        self.token = os.getenv('SWM_TOKEN', '')
        self.session = requests.Session()
        self.auth_token = None
        
        logger.info(f"VehicleService initialized with base URL: {self.base_url}")
        logger.info(f"Auth available: {'API Key' if self.api_key else ''} {'Username/Password' if self.username and self.password else ''} {'Token' if self.token else ''}")
    
    def get_live_vehicles(self) -> pd.DataFrame:
        """Fetch live vehicle data from SWM API."""
        try:
            # Get fresh token if needed
            if not self.token and self.username and self.password:
                self.auth_token = self._get_login_token()
            
            # Use the correct vehicle endpoint with pagination
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            
            endpoint = f'/api/vehicles/paginated?date={today}&size=542&sortBy=vehicleNo'
            url = f"{self.base_url}{endpoint}"
            
            logger.info(f"Fetching vehicles from: {url}")
            
            # Use bearer token
            headers = {'accept': '*/*'}
            token_to_use = self.token if self.token else self.auth_token
            
            if token_to_use:
                headers['Authorization'] = f'Bearer {token_to_use}'
                logger.info("Using bearer token")
            else:
                logger.warning("No bearer token available")
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                vehicles_data = response.json()
                logger.success(f"Successfully fetched vehicle data")
                return self._process_vehicle_data(vehicles_data)
            else:
                logger.error(f"API returned status {response.status_code}: {response.text[:200]}")
                return self._create_fallback_vehicles()
            

            
        except Exception as e:
            logger.error(f"Error fetching live vehicle data: {e}")
            return self._create_fallback_vehicles()
    
    def get_vehicles_by_ward(self, ward_no: str) -> pd.DataFrame:
        """Get vehicles filtered by ward number."""
        try:
            # Get all vehicles first
            all_vehicles = self.get_live_vehicles()
            
            # Filter by ward - check multiple possible ward field names
            ward_fields = ['ward', 'wardNo', 'ward_no', 'wardNumber', 'zone', 'area']
            
            filtered_vehicles = None
            for field in ward_fields:
                if field in all_vehicles.columns:
                    filtered_vehicles = all_vehicles[all_vehicles[field].astype(str) == str(ward_no)]
                    if len(filtered_vehicles) > 0:
                        logger.info(f"Found {len(filtered_vehicles)} vehicles in ward {ward_no} using field '{field}'")
                        break
            
            # If no ward field found or no matches, return active vehicles
            if filtered_vehicles is None or len(filtered_vehicles) == 0:
                logger.warning(f"No ward field found or no vehicles in ward {ward_no}, using all active vehicles")
                filtered_vehicles = all_vehicles[all_vehicles['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])]
            
            return filtered_vehicles
            
        except Exception as e:
            logger.error(f"Error filtering vehicles by ward {ward_no}: {e}")
            return self._create_fallback_vehicles()
    
    def _get_auth_methods(self):
        """Get all possible authentication methods to try."""
        methods = [{}]  # No auth first
        
        # Bearer token from env
        if self.token:
            methods.append({'headers': {'Authorization': f'Bearer {self.token}'}})
        
        # API key variations
        if self.api_key:
            methods.extend([
                {'headers': {'Authorization': f'Bearer {self.api_key}'}},
                {'headers': {'X-API-Key': self.api_key}},
                {'headers': {'api-key': self.api_key}},
                {'params': {'api_key': self.api_key}}
            ])
        
        # Basic auth
        if self.username and self.password:
            methods.append({'auth': (self.username, self.password)})
        
        # Use auth token if available
        if self.auth_token:
            methods.append({'headers': {'Authorization': f'Bearer {self.auth_token}'}})
        
        return methods
    
    def _get_login_token(self):
        """Get authentication token via login."""
        try:
            url = f"{self.base_url}/auth/login"
            login_data = {
                'loginId': self.username,
                'password': self.password
            }
            
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Attempting login to {url}")
            response = requests.post(url, json=login_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.success(f"Login successful")
                
                # Look for token in various fields
                token_fields = ['token', 'access_token', 'accessToken', 'authToken', 'jwt', 'bearerToken']
                for field in token_fields:
                    if field in data:
                        logger.success(f"Got auth token")
                        return data[field]
                    elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict) and field in data['data']:
                        logger.success(f"Got auth token from data")
                        return data['data'][field]
                
                logger.warning(f"Token not found. Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
                return None
            else:
                logger.error(f"Login failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return None
    
    def _process_vehicle_data(self, vehicles_data) -> pd.DataFrame:
        """Process and standardize vehicle data from API response."""
        # Convert to DataFrame - handle paginated response
        if isinstance(vehicles_data, list):
            df = pd.DataFrame(vehicles_data)
        elif isinstance(vehicles_data, dict):
            if 'content' in vehicles_data:  # Paginated response
                df = pd.DataFrame(vehicles_data['content'])
            elif 'data' in vehicles_data:
                df = pd.DataFrame(vehicles_data['data'])
            elif 'vehicles' in vehicles_data:
                df = pd.DataFrame(vehicles_data['vehicles'])
            else:
                df = pd.DataFrame([vehicles_data])
        else:
            return self._create_fallback_vehicles()
        
        df = self._standardize_vehicle_data(df)
        
        # Filter active vehicles - handle different status formats
        if 'status' in df.columns:
            active_vehicles = df[df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'OPERATIONAL'])].copy()
        else:
            # If no status column, assume all are active
            df['status'] = 'active'
            active_vehicles = df.copy()
        
        if len(active_vehicles) == 0:
            active_vehicles = df.copy()
        
        logger.success(f"Loaded {len(active_vehicles)} active vehicles from live API")
        return active_vehicles
    
    def _standardize_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize vehicle data column names and format."""
        # Common column mappings
        column_mappings = {
            'id': 'vehicle_id',
            'vehicleId': 'vehicle_id',
            'vehicleNo': 'vehicle_id',
            'vehicle_number': 'vehicle_id',
            'registration_number': 'vehicle_id',
            'wardNo': 'ward_no',
            'wardNumber': 'ward_no',
            'name': 'vehicle_name',
            'vehicleName': 'vehicle_name',
            'type': 'vehicle_type',
            'vehicleType': 'vehicle_type',
            'capacity': 'capacity',
            'vehicleCapacity': 'capacity',
            'location': 'location',
            'currentLocation': 'location',
            'latitude': 'lat',
            'longitude': 'lon',
            'lng': 'lon'
        }
        
        # Rename columns
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['vehicle_id', 'status']
        for col in required_columns:
            if col not in df.columns:
                if col == 'vehicle_id':
                    df['vehicle_id'] = [f"vehicle_{i+1}" for i in range(len(df))]
                elif col == 'status':
                    df['status'] = 'active'
        
        # Add default values for missing columns
        if 'vehicle_type' not in df.columns:
            df['vehicle_type'] = 'garbage_truck'
        if 'capacity' not in df.columns:
            df['capacity'] = 1000  # Default capacity in kg
        if 'ward_no' not in df.columns:
            df['ward_no'] = '1'  # Default ward
        
        return df
    
    def _create_fallback_vehicles(self) -> pd.DataFrame:
        """Create fallback vehicle data when API is unavailable."""
        logger.warning("Creating fallback vehicle data")
        
        fallback_data = [
            {'vehicle_id': 'SWM001', 'status': 'active', 'vehicle_type': 'garbage_truck', 'capacity': 1000},
            {'vehicle_id': 'SWM002', 'status': 'active', 'vehicle_type': 'garbage_truck', 'capacity': 1000},
            {'vehicle_id': 'SWM003', 'status': 'active', 'vehicle_type': 'garbage_truck', 'capacity': 1000},
            {'vehicle_id': 'SWM004', 'status': 'active', 'vehicle_type': 'garbage_truck', 'capacity': 1000},
            {'vehicle_id': 'SWM005', 'status': 'active', 'vehicle_type': 'garbage_truck', 'capacity': 1000}
        ]
        
        return pd.DataFrame(fallback_data)
    
    def get_vehicle_by_id(self, vehicle_id: str) -> Optional[Dict]:
        """Get specific vehicle data by ID."""
        try:
            url = f"{self.base_url}/api/vehicles/{vehicle_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Vehicle {vehicle_id} not found: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching vehicle {vehicle_id}: {e}")
            return None
    
    def update_vehicle_status(self, vehicle_id: str, status: str) -> bool:
        """Update vehicle status via API."""
        try:
            url = f"{self.base_url}/api/vehicles/{vehicle_id}/status"
            data = {'status': status}
            
            response = self.session.put(url, json=data, timeout=30)
            
            if response.status_code in [200, 204]:
                logger.info(f"Updated vehicle {vehicle_id} status to {status}")
                return True
            else:
                logger.error(f"Failed to update vehicle status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating vehicle status: {e}")
            return False