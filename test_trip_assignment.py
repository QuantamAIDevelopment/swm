"""Test script for trip assignment functionality."""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from clustering.trip_assignment import TripAssignmentManager
from configurations.config import Config

def create_test_buildings(num_houses: int) -> gpd.GeoDataFrame:
    """Create test building data."""
    data = []
    for i in range(num_houses):
        data.append({
            'geometry': Point(i * 0.001, i * 0.001),
            'road_node': (i * 0.001, i * 0.001),
            'snap_distance': 10.0
        })
    
    return gpd.GeoDataFrame(data)

def test_single_trip_scenario():
    """Test scenario where all houses fit in one trip."""
    print("=== Testing Single Trip Scenario ===")
    
    # 3 vehicles, 1200 houses (3 * 500 = 1500 capacity, so 1 trip needed)
    num_vehicles = 3
    num_houses = 1200
    
    buildings = create_test_buildings(num_houses)
    trip_manager = TripAssignmentManager()
    
    assignments = trip_manager.assign_trips(buildings, num_vehicles)
    
    print(f"Houses: {num_houses}, Vehicles: {num_vehicles}")
    print(f"Expected trips: 1, Actual trips: {assignments['num_trips']}")
    print(f"Houses per trip capacity: {assignments['houses_per_trip']}")
    
    # Validate
    assert assignments['num_trips'] == 1, "Should need only 1 trip"
    assert trip_manager.validate_no_overlap(assignments), "No overlap validation failed"
    
    print("Single trip scenario passed\n")

def test_multiple_trip_scenario():
    """Test scenario where multiple trips are needed."""
    print("=== Testing Multiple Trip Scenario ===")
    
    # 2 vehicles, 1200 houses (2 * 500 = 1000 capacity per trip, so 2 trips needed)
    num_vehicles = 2
    num_houses = 1200
    
    buildings = create_test_buildings(num_houses)
    trip_manager = TripAssignmentManager()
    
    assignments = trip_manager.assign_trips(buildings, num_vehicles)
    
    print(f"Houses: {num_houses}, Vehicles: {num_vehicles}")
    print(f"Expected trips: 2, Actual trips: {assignments['num_trips']}")
    print(f"Houses per trip capacity: {assignments['houses_per_trip']}")
    
    # Validate
    assert assignments['num_trips'] == 2, "Should need 2 trips"
    assert trip_manager.validate_no_overlap(assignments), "No overlap validation failed"
    
    # Check trip distribution
    trip1_houses = sum(len(v['buildings']) for v in assignments['assignments'][1].values())
    trip2_houses = sum(len(v['buildings']) for v in assignments['assignments'][2].values())
    
    print(f"Trip 1 houses: {trip1_houses}, Trip 2 houses: {trip2_houses}")
    assert trip1_houses + trip2_houses == num_houses, "Total houses mismatch"
    
    print("Multiple trip scenario passed\n")

def test_equal_distribution():
    """Test that houses are distributed equally among vehicles."""
    print("=== Testing Equal Distribution ===")
    
    num_vehicles = 3
    num_houses = 900  # Should distribute evenly: 300 per vehicle
    
    buildings = create_test_buildings(num_houses)
    trip_manager = TripAssignmentManager()
    
    assignments = trip_manager.assign_trips(buildings, num_vehicles)
    
    # Check distribution within first trip
    trip1_assignments = assignments['assignments'][1]
    house_counts = [v['house_count'] for v in trip1_assignments.values()]
    
    print(f"Houses per vehicle: {house_counts}")
    
    # Should be roughly equal (difference of at most 1)
    max_diff = max(house_counts) - min(house_counts)
    assert max_diff <= 1, f"Uneven distribution: max difference {max_diff}"
    
    print("Equal distribution test passed\n")

def test_configuration():
    """Test configuration values."""
    print("=== Testing Configuration ===")
    
    print(f"Houses per vehicle per trip: {Config.HOUSES_PER_VEHICLE_PER_TRIP}")
    print(f"Max trips per day: {Config.MAX_TRIPS_PER_DAY}")
    
    assert Config.HOUSES_PER_VEHICLE_PER_TRIP == 500, "Houses per trip should be 500"
    assert Config.MAX_TRIPS_PER_DAY == 2, "Max trips should be 2"
    
    print("Configuration test passed\n")

if __name__ == "__main__":
    print("Running Trip Assignment Tests\n")
    
    test_configuration()
    test_single_trip_scenario()
    test_multiple_trip_scenario()
    test_equal_distribution()
    
    print("All tests passed!")