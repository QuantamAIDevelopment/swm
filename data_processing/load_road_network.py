"""Load road network from GeoJSON and build NetworkX graph."""
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from loguru import logger
import numpy as np

class RoadNetworkLoader:
    def __init__(self):
        self.graph = nx.Graph()
        self.road_gdf = None
        
    def load_geojson(self, geojson_path: str) -> gpd.GeoDataFrame:
        """Load road network from GeoJSON file."""
        try:
            self.road_gdf = gpd.read_file(geojson_path)
            logger.info(f"Loaded {len(self.road_gdf)} road segments from {geojson_path}")
            return self.road_gdf
        except Exception as e:
            logger.error(f"Failed to load road network: {e}")
            raise
    
    def build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph from road network."""
        if self.road_gdf is None:
            raise ValueError("Road network not loaded")
            
        for idx, row in self.road_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    start_node = coords[i]
                    end_node = coords[i + 1]
                    
                    # Calculate edge weight (distance)
                    distance = Point(start_node).distance(Point(end_node))
                    
                    self.graph.add_edge(
                        start_node, 
                        end_node,
                        weight=distance,
                        geometry=LineString([start_node, end_node]),
                        road_id=idx
                    )
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_nearest_node(self, point: Point) -> tuple:
        """Find nearest node in the road network."""
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.graph.nodes():
            node_point = Point(node)
            distance = point.distance(node_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
                
        return nearest_node, min_distance