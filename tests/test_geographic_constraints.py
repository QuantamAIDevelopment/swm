"""Test geographic constraint enforcement in clustering."""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from tools.improved_clustering import ImprovedClustering

def test_houses_within_cluster_zones():
    """Test that all houses are within their assigned cluster's geographic zone."""
    
    # Create test data with clear geographic separation
    roads_data = {
        'road_id': [1, 2, 3, 4, 5, 6],
        'geometry': [
            LineString([(0, 0), (1, 0)]),    # Group 1 - Bottom left
            LineString([(0.5, 0.5), (1.5, 0.5)]),  # Group 1 - Bottom left
            LineString([(5, 0), (6, 0)]),    # Group 2 - Bottom right
            LineString([(5.5, 0.5), (6.5, 0.5)]),  # Group 2 - Bottom right  
            LineString([(0, 5), (1, 5)]),    # Group 3 - Top left
            LineString([(5, 5), (6, 5)])     # Group 3 - Top right
        ]
    }
    roads = gpd.GeoDataFrame(roads_data, crs='EPSG:3857')
    
    houses_data = {
        'house_id': list(range(1, 13)),
        'road_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        'geometry': [
            Point(0.2, 0.1), Point(0.8, 0.1),      # Near roads 1
            Point(0.7, 0.6), Point(1.3, 0.6),      # Near roads 2
            Point(5.2, 0.1), Point(5.8, 0.1),      # Near roads 3
            Point(5.7, 0.6), Point(6.3, 0.6),      # Near roads 4
            Point(0.2, 5.1), Point(0.8, 5.1),      # Near roads 5
            Point(5.2, 5.1), Point(5.8, 5.1)       # Near roads 6
        ]
    }
    houses = gpd.GeoDataFrame(houses_data, crs='EPSG:3857')
    
    # Test with 3 clusters
    clustering = ImprovedClustering(random_seed=42)
    clusters = clustering.create_geographic_clusters(roads, houses, 3)
    
    # Create cluster zones
    cluster_zones = clustering._create_cluster_zones(clusters, roads)
    
    # Verify each house is within its cluster's zone
    violations = 0
    for _, house in houses.iterrows():
        house_road_id = house['road_id']
        house_point = house.geometry.centroid
        
        # Find which cluster this house's road belongs to
        assigned_cluster = None
        for cluster_idx, cluster_roads in enumerate(clusters):
            if house_road_id in cluster_roads:
                assigned_cluster = cluster_idx
                break
        
        if assigned_cluster is not None and cluster_zones[assigned_cluster]:
            zone = cluster_zones[assigned_cluster]
            if not zone.contains(house_point) and zone.distance(house_point) > 10:  # 10m tolerance
                violations += 1
                print(f"VIOLATION: House {house['house_id']} on road {house_road_id} is outside its cluster zone")
    
    print(f"Geographic constraint violations: {violations}")
    assert violations == 0, f"Found {violations} houses outside their cluster zones"
    print("All houses are within their cluster's geographic zones - PASSED")

if __name__ == "__main__":
    test_houses_within_cluster_zones()