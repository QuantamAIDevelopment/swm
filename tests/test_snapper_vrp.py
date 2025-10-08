"""Unit tests for road snapper and VRP solver."""
import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from tools.road_snapper import RoadSnapper
from tools.vrp_solver import VRPSolver

class TestRoadSnapper:
    def setup_method(self):
        """Set up test data."""
        # Create simple road network
        roads_data = [
            {'road_id': 0, 'geometry': LineString([(0, 0), (10, 0)])},
            {'road_id': 1, 'geometry': LineString([(10, 0), (10, 10)])},
            {'road_id': 2, 'geometry': LineString([(10, 10), (0, 10)])},
            {'road_id': 3, 'geometry': LineString([(0, 10), (0, 0)])}
        ]
        self.road_network = gpd.GeoDataFrame(roads_data, crs="EPSG:3857")
        
        # Create houses near roads
        houses_data = [
            {'house_id': 'h1', 'geometry': Point(2, 1)},   # Near road 0
            {'house_id': 'h2', 'geometry': Point(9, 2)},   # Near road 1
            {'house_id': 'h3', 'geometry': Point(8, 9)},   # Near road 2
            {'house_id': 'h4', 'geometry': Point(1, 8)}    # Near road 3
        ]
        self.houses = gpd.GeoDataFrame(houses_data, crs="EPSG:3857")
        
        self.snapper = RoadSnapper(self.road_network)
    
    def test_snap_houses_to_roads(self):
        """Test house snapping functionality."""
        snapped_houses, segment_counts = self.snapper.snap_houses_to_roads(self.houses)
        
        # Check that all houses are snapped
        assert len(snapped_houses) == len(self.houses)
        assert all(snapped_houses['road_id'].notna())
        
        # Check segment counts
        assert len(segment_counts) <= len(self.road_network)
        assert sum(segment_counts.values()) == len(self.houses)
    
    def test_get_road_graph_nodes(self):
        """Test node extraction for VRP graph."""
        snapped_houses, _ = self.snapper.snap_houses_to_roads(self.houses)
        nodes = self.snapper.get_road_graph_nodes(snapped_houses)
        
        # Should have road endpoints + house nodes
        expected_road_nodes = len(self.road_network) * 2  # Start and end for each road
        expected_house_nodes = len(self.houses)
        
        assert len(nodes) == expected_road_nodes + expected_house_nodes
        assert 'node_type' in nodes.columns

class TestVRPSolver:
    def setup_method(self):
        """Set up test data."""
        # Create simple road network
        roads_data = [
            {'road_id': 0, 'geometry': LineString([(0, 0), (10, 0)])},
            {'road_id': 1, 'geometry': LineString([(10, 0), (10, 10)])},
            {'road_id': 2, 'geometry': LineString([(10, 10), (0, 10)])},
            {'road_id': 3, 'geometry': LineString([(0, 10), (0, 0)])}
        ]
        self.road_network = gpd.GeoDataFrame(roads_data, crs="EPSG:3857")
        
        # Create cluster nodes
        nodes_data = [
            {'node_id': 'n1', 'geometry': Point(2, 2), 'node_type': 'house'},
            {'node_id': 'n2', 'geometry': Point(8, 2), 'node_type': 'house'},
            {'node_id': 'n3', 'geometry': Point(8, 8), 'node_type': 'house'},
            {'node_id': 'n4', 'geometry': Point(2, 8), 'node_type': 'house'}
        ]
        self.cluster_nodes = gpd.GeoDataFrame(nodes_data, crs="EPSG:3857")
        
        self.vrp_solver = VRPSolver(self.road_network)
    
    def test_build_road_graph(self):
        """Test road graph construction."""
        graph = self.vrp_solver.graph
        
        # Should have nodes and edges
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
    
    def test_solve_vrp(self):
        """Test VRP solving."""
        route = self.vrp_solver.solve_vrp(self.cluster_nodes)
        
        # Should return a valid route
        assert route is not None
        assert len(route) == len(self.cluster_nodes) + 1  # Include return to start
        assert route[0] == 0  # Starts at index 0
    
    def test_build_route_geometry(self):
        """Test route geometry construction."""
        route_order = [0, 1, 2, 3, 0]
        geometry = self.vrp_solver.build_route_geometry(route_order, self.cluster_nodes)
        
        assert isinstance(geometry, LineString)
        assert len(geometry.coords) >= 2

class TestIntegration:
    def test_end_to_end_small_dataset(self):
        """End-to-end test with synthetic small dataset."""
        # Create ward boundary
        ward_boundary = gpd.GeoDataFrame([{
            'ward_id': 1,
            'geometry': Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)])
        }], crs="EPSG:3857")
        
        # Create road network (grid)
        roads = []
        road_id = 0
        # Horizontal roads
        for y in [5, 10, 15]:
            roads.append({
                'road_id': road_id,
                'geometry': LineString([(0, y), (20, y)])
            })
            road_id += 1
        
        # Vertical roads
        for x in [5, 10, 15]:
            roads.append({
                'road_id': road_id,
                'geometry': LineString([(x, 0), (x, 20)])
            })
            road_id += 1
        
        road_network = gpd.GeoDataFrame(roads, crs="EPSG:3857")
        
        # Create houses (60 houses in grid pattern)
        houses = []
        house_id = 0
        for x in range(2, 19, 2):
            for y in range(2, 19, 2):
                houses.append({
                    'house_id': f'h{house_id}',
                    'geometry': Point(x, y)
                })
                house_id += 1
        
        houses_gdf = gpd.GeoDataFrame(houses, crs="EPSG:3857")
        
        # Create vehicles
        vehicles = pd.DataFrame([
            {'vehicle_id': 'v1', 'vehicle_type': 'truck', 'ward_no': 1, 'status': 'active'},
            {'vehicle_id': 'v2', 'vehicle_type': 'truck', 'ward_no': 1, 'status': 'active'},
            {'vehicle_id': 'v3', 'vehicle_type': 'truck', 'ward_no': 1, 'status': 'active'}
        ])
        
        # Test snapping
        snapper = RoadSnapper(road_network)
        snapped_houses, segment_counts = snapper.snap_houses_to_roads(houses_gdf)
        
        # Verify each house is assigned to exactly one road segment
        assert len(snapped_houses) == len(houses_gdf)
        assert all(snapped_houses['road_id'].notna())
        
        # Test VRP solving
        vrp_solver = VRPSolver(road_network)
        
        # Create a small cluster for testing
        test_cluster = snapped_houses.head(10)  # First 10 houses
        cluster_nodes = snapper.get_road_graph_nodes(test_cluster)
        
        route = vrp_solver.solve_vrp(cluster_nodes)
        
        # Verify route properties
        assert route is not None
        assert len(route) > 0
        
        # Test route geometry
        geometry = vrp_solver.build_route_geometry(route, cluster_nodes)
        assert isinstance(geometry, LineString)
        assert geometry.length > 0
        
        print("End-to-end test passed successfully!")
        print(f"- Processed {len(houses_gdf)} houses")
        print(f"- Snapped to {len(segment_counts)} road segments")
        print(f"- Generated route with {len(route)} stops")
        print(f"- Route length: {geometry.length:.2f} units")