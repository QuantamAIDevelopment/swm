"""Snap building coordinates to nearest road nodes."""
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from loguru import logger
import pandas as pd

class BuildingSnapper:
    def __init__(self, road_graph: nx.Graph):
        self.road_graph = road_graph
        
    def load_buildings(self, geojson_path: str) -> gpd.GeoDataFrame:
        """Load buildings from GeoJSON file."""
        try:
            buildings_gdf = gpd.read_file(geojson_path)
            logger.info(f"Loaded {len(buildings_gdf)} buildings from {geojson_path}")
            return buildings_gdf
        except Exception as e:
            logger.error(f"Failed to load buildings: {e}")
            raise
    
    def snap_to_road_network(self, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Snap building coordinates to nearest road nodes."""
        snapped_buildings = []
        
        for idx, building in buildings_gdf.iterrows():
            if building.geometry.geom_type == 'Point':
                building_point = building.geometry
            else:
                # Use centroid for polygons
                building_point = building.geometry.centroid
            
            nearest_node, distance = self._find_nearest_node(building_point)
            
            snapped_building = building.copy()
            snapped_building['original_geometry'] = building.geometry
            snapped_building['geometry'] = Point(nearest_node)
            snapped_building['snap_distance'] = distance
            snapped_building['road_node'] = nearest_node
            
            snapped_buildings.append(snapped_building)
        
        result_gdf = gpd.GeoDataFrame(snapped_buildings)
        logger.info(f"Snapped {len(result_gdf)} buildings to road network")
        return result_gdf
    
    def _find_nearest_node(self, point: Point) -> tuple:
        """Find nearest node in road graph."""
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.road_graph.nodes():
            node_point = Point(node)
            distance = point.distance(node_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
                
        return nearest_node, min_distance