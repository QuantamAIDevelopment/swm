"""Fixed route assignment agent with spatial partitioning."""
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
from typing import List, Dict
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from models.blackboard_entry import RouteResult
from tools.road_snapper import RoadSnapper
from tools.improved_clustering import ImprovedClustering
from configurations.config import Config

logger = logging.getLogger(__name__)

class RouteAssignmentAgent:
    def __init__(self):
        self.random_seed = Config.RANDOM_SEED
        
    def process_data(self, data: Dict) -> List[RouteResult]:
        """Main processing pipeline with spatial partitioning."""
        logger.info("Processing route assignment with spatial partitioning")
        
        # Reproject to metric CRS
        ward_boundaries = data["ward_boundaries"].to_crs(Config.TARGET_CRS)
        road_network = data["road_network"].to_crs(Config.TARGET_CRS)
        houses = data["houses"].to_crs(Config.TARGET_CRS)
        vehicles = data["vehicles"]
        
        # Filter active vehicles
        active_vehicles = vehicles[vehicles['status'].str.upper() == 'ACTIVE'].copy()
        if len(active_vehicles) == 0:
            active_vehicles = vehicles.copy()
            logger.warning("No ACTIVE vehicles found, using all vehicles")
        
        logger.info(f"Found {len(active_vehicles)} active vehicles")
        
        # Snap houses to roads
        road_snapper = RoadSnapper(road_network)
        snapped_houses, _ = road_snapper.snap_houses_to_roads(houses)
        
        # Create geographic clusters ensuring houses are within zones
        clustering = ImprovedClustering(random_seed=self.random_seed)
        clusters = clustering.create_geographic_clusters(road_network, snapped_houses, len(active_vehicles))
        
        # Convert clusters to partitions format
        partitions = self._convert_clusters_to_partitions(clusters, snapped_houses, road_network)
        
        # Assign vehicles to partitions
        vehicle_assignments = self._assign_vehicles_to_partitions(active_vehicles, partitions)
        
        # Create non-overlapping routes
        routes = []
        for vehicle_id, partition_data in vehicle_assignments.items():
            route = self._create_partition_route(vehicle_id, partition_data, road_network)
            if route:
                routes.append(route)
        
        # Validate no overlaps
        self._validate_no_overlaps(routes, road_network)
        
        logger.info(f"Generated {len(routes)} non-overlapping routes")
        return routes
    
    def _convert_clusters_to_partitions(self, clusters: List[List], houses: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> List[Dict]:
        """Convert road clusters to partition format."""
        partitions = []
        
        for cluster_id, cluster_road_ids in enumerate(clusters):
            if not cluster_road_ids:
                continue
                
            # Get roads in this cluster
            cluster_roads = roads[roads['road_id'].isin(cluster_road_ids)]
            
            # Get houses on these roads
            cluster_houses = houses[houses['road_id'].isin(cluster_road_ids)]
            
            if len(cluster_houses) > 0:
                # Create convex hull as partition polygon
                cluster_geom = unary_union(cluster_roads.geometry)
                partition_poly = cluster_geom.convex_hull.buffer(50)  # 50m buffer
                
                partitions.append({
                    'id': cluster_id,
                    'polygon': partition_poly,
                    'houses': cluster_houses,
                    'roads': cluster_roads,
                    'road_ids': set(cluster_road_ids)
                })
        
        logger.info(f"Created {len(partitions)} geographic partitions")
        return partitions
    
    def _assign_vehicles_to_partitions(self, vehicles: pd.DataFrame, partitions: List[Dict]) -> Dict[str, Dict]:
        """Assign vehicles to partitions."""
        assignments = {}
        
        # Sort partitions by house count (largest first)
        partitions_sorted = sorted(partitions, key=lambda p: len(p['houses']), reverse=True)
        
        for i, (_, vehicle) in enumerate(vehicles.iterrows()):
            if i < len(partitions_sorted):
                partition = partitions_sorted[i]
                assignments[vehicle['vehicle_id']] = partition
                logger.info(f"Vehicle {vehicle['vehicle_id']} assigned to partition {partition['id']} with {len(partition['houses'])} houses")
        
        return assignments
    
    def _create_partition_route(self, vehicle_id: str, partition_data: Dict, road_network: gpd.GeoDataFrame) -> RouteResult:
        """Create route for a partition using only roads within that partition."""
        houses = partition_data['houses']
        partition_roads = partition_data['roads']
        
        if len(houses) == 0 or len(partition_roads) == 0:
            return None
        
        house_ids = [str(house.get('house_id', idx)) for idx, house in houses.iterrows()]
        road_segment_ids = [str(idx) for idx in partition_roads.index]
        
        # Create route geometry from partition roads
        try:
            road_geoms = []
            for _, road in partition_roads.iterrows():
                geom = road.geometry
                if geom.geom_type == 'LineString':
                    road_geoms.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    road_geoms.extend(list(geom.geoms))
            
            if road_geoms:
                # Union all road segments in this partition
                route_geometry = unary_union(road_geoms)
                
                # Convert to LineString if needed
                if route_geometry.geom_type == 'MultiLineString':
                    coords = []
                    for line in route_geometry.geoms:
                        coords.extend(list(line.coords))
                    route_geometry = LineString(coords)
                elif route_geometry.geom_type != 'LineString':
                    # Fallback: create line from house centroids
                    centroids = [(h.geometry.centroid.x, h.geometry.centroid.y) for _, h in houses.iterrows()]
                    if len(centroids) >= 2:
                        route_geometry = LineString(centroids)
                    else:
                        return None
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Route creation failed for {vehicle_id}: {e}")
            return None
        
        return RouteResult(
            vehicle_id=vehicle_id,
            route_id=f"partition_route_{vehicle_id}",
            ordered_house_ids=house_ids,
            road_segment_ids=road_segment_ids,
            start_node=f"start_{vehicle_id}",
            end_node=f"end_{vehicle_id}",
            total_distance_meters=route_geometry.length,
            status="active",
            geometry=route_geometry
        )
    
    def _validate_no_overlaps(self, routes: List[RouteResult], road_network: gpd.GeoDataFrame):
        """Validate that there are no overlapping road segments."""
        all_assigned_segments = set()
        overlap_count = 0
        
        for route in routes:
            route_segments = set(route.road_segment_ids)
            overlaps = all_assigned_segments.intersection(route_segments)
            
            if overlaps:
                overlap_count += len(overlaps)
                logger.error(f"Vehicle {route.vehicle_id} has {len(overlaps)} overlapping segments")
            
            all_assigned_segments.update(route_segments)
        
        total_segments = len(road_network)
        coverage_percent = (len(all_assigned_segments) / total_segments) * 100
        
        logger.info(f"Coverage: {coverage_percent:.1f}% ({len(all_assigned_segments)}/{total_segments} segments)")
        
        if overlap_count == 0:
            logger.info("✅ ZERO OVERLAPS - Perfect spatial partitioning")
        else:
            logger.error(f"❌ {overlap_count} overlapping segments found")
    
    def reassign_vehicle_routes(self, upload_id: str, unavailable_vehicle_id: str) -> List[RouteResult]:
        """Reassign routes when a vehicle becomes unavailable."""
        logger.info(f"Reassigning routes after vehicle {unavailable_vehicle_id} became unavailable")
        return []  # Simplified for now