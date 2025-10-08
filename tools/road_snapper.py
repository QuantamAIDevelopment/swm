"""Tool for snapping houses to road segments."""
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import numpy as np
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class RoadSnapper:
    def __init__(self, road_network: gpd.GeoDataFrame):
        self.road_network = road_network.copy()
        # Ensure road_id exists and is properly indexed
        if 'road_id' not in self.road_network.columns:
            self.road_network['road_id'] = self.road_network.index
        # Reset index to ensure road_id is accessible
        self.road_network = self.road_network.reset_index(drop=True)
    
    def snap_houses_to_roads(self, houses: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
        """
        Snap houses to nearest road segments and count houses per segment.
        
        Returns:
            Tuple of (snapped_houses_gdf, segment_house_counts)
        """
        logger.info(f"Snapping {len(houses)} houses to {len(self.road_network)} road segments")
        
        # Ensure same CRS
        if houses.crs != self.road_network.crs:
            houses = houses.to_crs(self.road_network.crs)
        
        snapped_houses = houses.copy()
        snapped_houses['road_id'] = None
        snapped_houses['snap_distance'] = None
        snapped_houses['snapped_geometry'] = None
        
        segment_house_counts = {}
        
        for idx, house in houses.iterrows():
            house_point = house.geometry
            min_distance = float('inf')
            closest_road_id = None
            snapped_point = None
            
            # Find nearest road segment
            for road_idx, road in self.road_network.iterrows():
                road_geom = road.geometry
                if road_geom is None or road_geom.is_empty:
                    continue
                    
                # Get nearest point on road to house
                try:
                    nearest_pt = nearest_points(house_point, road_geom)[1]
                    distance = house_point.distance(nearest_pt)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_road_id = road['road_id']
                        snapped_point = nearest_pt
                except Exception as e:
                    logger.warning(f"Error snapping house {idx} to road {road_idx}: {e}")
                    continue
            
            snapped_houses.at[idx, 'road_id'] = closest_road_id
            snapped_houses.at[idx, 'snap_distance'] = min_distance
            snapped_houses.at[idx, 'snapped_geometry'] = snapped_point
            
            # Count houses per segment
            if closest_road_id not in segment_house_counts:
                segment_house_counts[closest_road_id] = 0
            segment_house_counts[closest_road_id] += 1
        
        logger.info(f"Snapped houses to {len(segment_house_counts)} unique road segments")
        return snapped_houses, segment_house_counts
    
    def get_road_graph_nodes(self, snapped_houses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Extract nodes from road network for VRP graph construction.
        """
        nodes = []
        node_id = 0
        
        # Add road segment endpoints as nodes
        for idx, road in self.road_network.iterrows():
            if isinstance(road.geometry, LineString):
                coords = list(road.geometry.coords)
                # Start point
                nodes.append({
                    'node_id': f"node_{node_id}",
                    'geometry': Point(coords[0]),
                    'road_id': road['road_id'],
                    'node_type': 'road_start'
                })
                node_id += 1
                
                # End point
                nodes.append({
                    'node_id': f"node_{node_id}",
                    'geometry': Point(coords[-1]),
                    'road_id': road['road_id'],
                    'node_type': 'road_end'
                })
                node_id += 1
        
        # Add snapped house locations as nodes
        for idx, house in snapped_houses.iterrows():
            nodes.append({
                'node_id': f"house_{house.get('house_id', idx)}",
                'geometry': house['snapped_geometry'],
                'road_id': house['road_id'],
                'node_type': 'house',
                'house_id': house.get('house_id', idx)
            })
        
        nodes_gdf = gpd.GeoDataFrame(nodes, crs=self.road_network.crs)
        logger.info(f"Created {len(nodes_gdf)} nodes for VRP graph")
        return nodes_gdf