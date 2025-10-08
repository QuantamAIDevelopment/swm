"""Test improved clustering functionality."""
import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from tools.improved_clustering import ImprovedClustering

def test_geographic_clustering():
    """Test that houses are assigned within their cluster's geographic zone."""
    
    # Create test data
    roads_data = {
        'road_id': [1, 2, 3, 4],
        'geometry': [
            LineString([(0, 0), (1, 0)]),    # Bottom left
            LineString([(2, 0), (3, 0)]),    # Bottom right  
            LineString([(0, 2), (1, 2)]),    # Top left
            LineString([(2, 2), (3, 2)])     # Top right
        ]
    }
    roads = gpd.GeoDataFrame(roads_data, crs='EPSG:4326')
    
    houses_data = {
        'house_id': [1, 2, 3, 4, 5, 6],
        'road_id': [1, 1, 2, 2, 3, 4],
        'geometry': [
            Point(0.5, 0.1),   # Near road 1
            Point(0.8, 0.1),   # Near road 1
            Point(2.5, 0.1),   # Near road 2
            Point(2.8, 0.1),   # Near road 2
            Point(0.5, 2.1),   # Near road 3
            Point(2.5, 2.1)    # Near road 4
        ]
    }
    houses = gpd.GeoDataFrame(houses_data, crs='EPSG:4326')
    
    # Test clustering
    clustering = ImprovedClustering(random_seed=42)
    clusters = clustering.create_geographic_clusters(roads, houses, 2)
    
    # Verify results
    assert len(clusters) == 2
    assert all(len(cluster) > 0 for cluster in clusters)
    
    # Check that all roads are assigned
    all_assigned_roads = set()
    for cluster in clusters:
        all_assigned_roads.update(cluster)
    
    assert len(all_assigned_roads) == len(roads)
    
    print("✅ Geographic clustering test passed")

if __name__ == "__main__":
    test_geographic_clustering()