"""Trip assignment logic for vehicles based on capacity constraints."""
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple
from loguru import logger
from configurations.config import Config

class TripAssignmentManager:
    def __init__(self):
        self.houses_per_trip = Config.HOUSES_PER_VEHICLE_PER_TRIP
        self.max_trips_per_day = Config.MAX_TRIPS_PER_DAY
    
    def assign_trips(self, clustered_buildings: gpd.GeoDataFrame, num_vehicles: int) -> Dict:
        """
        Assign trips based on vehicle capacity and house count.
        Returns trip assignments with no overlap between trips.
        """
        total_houses = len(clustered_buildings)
        total_capacity_single_trip = num_vehicles * self.houses_per_trip
        
        logger.info(f"Total houses: {total_houses}, Vehicle capacity (1 trip): {total_capacity_single_trip}")
        
        # Determine number of trips needed
        if total_houses <= total_capacity_single_trip:
            num_trips = 1
            logger.info("All houses can be covered in a single trip")
        else:
            num_trips = min(2, (total_houses + total_capacity_single_trip - 1) // total_capacity_single_trip)
            logger.info(f"Multiple trips needed: {num_trips}")
        
        # Assign houses to trips with no overlap
        trip_assignments = self._distribute_houses_to_trips(
            clustered_buildings, num_vehicles, num_trips
        )
        
        return {
            'num_trips': num_trips,
            'assignments': trip_assignments,
            'total_houses': total_houses,
            'houses_per_trip': self.houses_per_trip
        }
    
    def _distribute_houses_to_trips(self, buildings: gpd.GeoDataFrame, 
                                  num_vehicles: int, num_trips: int) -> Dict:
        """Distribute houses evenly across trips with no overlap."""
        total_houses = len(buildings)
        houses_per_trip = total_houses // num_trips
        remaining_houses = total_houses % num_trips
        
        trip_assignments = {}
        start_idx = 0
        
        for trip_num in range(1, num_trips + 1):
            # Calculate houses for this trip
            trip_house_count = houses_per_trip
            if trip_num <= remaining_houses:
                trip_house_count += 1
            
            end_idx = start_idx + trip_house_count
            trip_buildings = buildings.iloc[start_idx:end_idx].copy()
            
            # Redistribute buildings among vehicles for this trip
            trip_assignments[trip_num] = self._assign_vehicles_to_trip(
                trip_buildings, num_vehicles, trip_num
            )
            
            start_idx = end_idx
            logger.info(f"Trip {trip_num}: {len(trip_buildings)} houses assigned")
        
        return trip_assignments
    
    def _assign_vehicles_to_trip(self, trip_buildings: gpd.GeoDataFrame, 
                               num_vehicles: int, trip_num: int) -> Dict:
        """Assign vehicles to buildings within a single trip."""
        total_houses = len(trip_buildings)
        houses_per_vehicle = total_houses // num_vehicles
        remaining_houses = total_houses % num_vehicles
        
        vehicle_assignments = {}
        start_idx = 0
        
        for vehicle_id in range(num_vehicles):
            # Calculate houses for this vehicle
            vehicle_house_count = houses_per_vehicle
            if vehicle_id < remaining_houses:
                vehicle_house_count += 1
            
            if vehicle_house_count > 0:
                end_idx = start_idx + vehicle_house_count
                vehicle_buildings = trip_buildings.iloc[start_idx:end_idx].copy()
                
                # Add trip and vehicle metadata
                vehicle_buildings['trip_number'] = trip_num
                vehicle_buildings['vehicle_id'] = vehicle_id
                vehicle_buildings['cluster'] = f"trip_{trip_num}_vehicle_{vehicle_id}"
                
                vehicle_assignments[vehicle_id] = {
                    'buildings': vehicle_buildings,
                    'house_count': len(vehicle_buildings),
                    'trip_number': trip_num
                }
                
                start_idx = end_idx
        
        return vehicle_assignments
    
    def get_trip_summary(self, trip_assignments: Dict) -> pd.DataFrame:
        """Generate summary statistics for trip assignments."""
        summary_data = []
        
        for trip_num, trip_data in trip_assignments['assignments'].items():
            for vehicle_id, vehicle_data in trip_data.items():
                summary_data.append({
                    'trip_number': trip_num,
                    'vehicle_id': vehicle_id,
                    'house_count': vehicle_data['house_count'],
                    'capacity_utilization': vehicle_data['house_count'] / self.houses_per_trip
                })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"Generated trip summary for {len(summary_data)} vehicle-trip combinations")
        return summary_df
    
    def validate_no_overlap(self, trip_assignments: Dict) -> bool:
        """Validate that no house is assigned to multiple trips."""
        all_house_ids = set()
        
        for trip_num, trip_data in trip_assignments['assignments'].items():
            for vehicle_id, vehicle_data in trip_data.items():
                buildings = vehicle_data['buildings']
                house_ids = set(buildings.index)
                
                # Check for overlap
                overlap = all_house_ids.intersection(house_ids)
                if overlap:
                    logger.error(f"Overlap detected: {len(overlap)} houses assigned to multiple trips")
                    return False
                
                all_house_ids.update(house_ids)
        
        logger.info("No overlap detected - each house assigned to exactly one trip")
        return True