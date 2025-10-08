"""Route assignment agent for clustering and VRP coordination."""
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from datetime import datetime
from shapely.geometry import LineString
from shapely.ops import unary_union

from core.blackboard import blackboard
from models.blackboard_entry import UploadData, RouteResult
from tools.road_snapper import RoadSnapper
from tools.vrp_solver import VRPSolver
from tools.improved_clustering import ImprovedClustering
from configurations.config import Config

logger = logging.getLogger(__name__)

class RouteAssignmentAgent:
    def __init__(self):
        self.random_seed = Config.RANDOM_SEED
        
    def process_data(self, data: Dict) -> List[RouteResult]:
        """Main processing pipeline for route assignment."""
        logger.info("Processing route assignment")
        
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
        
        if len(active_vehicles) == 0:
            raise ValueError("No vehicles found in the dataset")
        
        logger.info(f"Found {len(active_vehicles)} active vehicles")
        
        # Snap houses to roads
        road_snapper = RoadSnapper(road_network)
        snapped_houses, segment_house_counts = road_snapper.snap_houses_to_roads(houses)
        
        # Create non-overlapping routes using road-first approach
        routes = self._create_road_based_routes(snapped_houses, road_network, active_vehicles)
        
        if not routes:
            logger.warning("No routes were generated")
            return []
        
        logger.info(f"Generated {len(routes)} non-overlapping routes")
        return routes
    
    def _create_road_based_routes(self, snapped_houses: gpd.GeoDataFrame, 
                                road_network: gpd.GeoDataFrame, 
                                vehicles: pd.DataFrame) -> List[RouteResult]:
        """Create routes by partitioning road network first, then assigning houses."""
        
        # Step 1: Ensure road_network has road_id column
        if 'road_id' not in road_network.columns:
            road_network = road_network.copy()
            road_network['road_id'] = road_network.index
        
        # Get ALL roads that have houses - ensure no road is missed
        roads_with_houses = set(snapped_houses['road_id'].dropna().unique())
        active_roads = road_network[road_network['road_id'].isin(roads_with_houses)].copy()
        
        if len(active_roads) == 0:
            logger.warning("No roads with houses found")
            return []
        
        logger.info(f"Found {len(active_roads)} roads with houses covering {len(snapped_houses)} houses")
        
        # Step 2: Create balanced non-overlapping road clusters
        n_vehicles = len(vehicles)
        road_clusters = self._create_balanced_road_clusters(active_roads, snapped_houses, n_vehicles)
        
        # Step 3: Create one route per vehicle from road clusters
        routes = []
        vehicle_list = vehicles['vehicle_id'].tolist()
        
        for cluster_id, cluster_road_ids in enumerate(road_clusters):
            if cluster_id >= len(vehicle_list) or not cluster_road_ids:
                continue
                
            vehicle_id = vehicle_list[cluster_id]
            
            # Get houses on these roads
            cluster_houses = snapped_houses[snapped_houses['road_id'].isin(cluster_road_ids)]
            
            # Get road geometries
            cluster_roads = active_roads[active_roads['road_id'].isin(cluster_road_ids)]
            
            # Create route geometry from roads
            route_geometry = self._create_route_geometry(cluster_roads)
            
            if route_geometry and len(cluster_houses) > 0:
                house_ids = [str(h.get('house_id', idx)) for idx, h in cluster_houses.iterrows()]
                
                route = RouteResult(
                    vehicle_id=vehicle_id,
                    route_id=f"route_{vehicle_id}_{cluster_id}",
                    ordered_house_ids=house_ids,
                    road_segment_ids=[str(rid) for rid in cluster_road_ids],
                    start_node=f"start_{vehicle_id}",
                    end_node=f"end_{vehicle_id}",
                    total_distance_meters=route_geometry.length,
                    status="active",
                    geometry=route_geometry
                )
                routes.append(route)
                
                logger.info(f"Vehicle {vehicle_id}: {len(cluster_road_ids)} roads, {len(cluster_houses)} houses")
        
        # Validate complete coverage and no overlaps
        self._validate_complete_coverage(routes, active_roads, snapped_houses)
        
        return routes
    
    def _create_balanced_road_clusters(self, roads: gpd.GeoDataFrame, 
                                     snapped_houses: gpd.GeoDataFrame, 
                                     n_clusters: int) -> List[List]:
        """Create balanced clusters ensuring houses are within geographic zones."""
        
        # Use improved clustering that enforces geographic constraints
        clustering = ImprovedClustering(random_seed=self.random_seed)
        clusters = clustering.create_geographic_clusters(roads, snapped_houses, n_clusters)
        
        # Fallback if improved clustering fails
        if not clusters or all(len(cluster) == 0 for cluster in clusters):
            logger.warning("Improved clustering failed, using fallback")
            return self._fallback_strip_clustering(roads, snapped_houses, n_clusters)
        
        return clusters
    
    def _fallback_strip_clustering(self, roads: gpd.GeoDataFrame, 
                                 snapped_houses: gpd.GeoDataFrame, 
                                 n_clusters: int) -> List[List]:
        """Fallback horizontal strip clustering."""
        all_road_ids = list(roads['road_id'].unique())
        
        # Get Y coordinates for horizontal strip clustering
        road_y_coords = []
        for road_id in all_road_ids:
            road_row = roads[roads['road_id'] == road_id].iloc[0]
            centroid = road_row.geometry.centroid
            road_y_coords.append((road_id, centroid.y))
        
        # Sort roads by Y coordinate (north to south)
        road_y_coords.sort(key=lambda x: x[1], reverse=True)
        
        # Create horizontal strips by dividing Y range
        y_values = [y_coord for _, y_coord in road_y_coords]
        y_min, y_max = min(y_values), max(y_values)
        strip_height = (y_max - y_min) / n_clusters
        
        # Assign roads to horizontal strips
        clusters = [[] for _ in range(n_clusters)]
        
        for road_id, y_coord in road_y_coords:
            strip_index = min(int((y_max - y_coord) / strip_height), n_clusters - 1)
            clusters[strip_index].append(road_id)
        
        return clusters
    
    def _fix_spatial_overlaps(self, clusters, roads, snapped_houses):
        """Fix spatial overlaps by reassigning roads based on house locations."""
        # Create mapping of road_id to cluster
        road_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters):
            for road_id in cluster:
                road_to_cluster[road_id] = cluster_idx
        
        # Check each house and its assigned road
        for _, house in snapped_houses.iterrows():
            house_road_id = house['road_id']
            if house_road_id not in road_to_cluster:
                continue
                
            house_cluster = road_to_cluster[house_road_id]
            house_geom = house.geometry
            
            # Check if house is spatially closer to a different cluster
            min_distance = float('inf')
            best_cluster = house_cluster
            
            for cluster_idx, cluster in enumerate(clusters):
                if cluster_idx == house_cluster or not cluster:
                    continue
                    
                # Calculate distance to cluster centroid
                cluster_roads = roads[roads['road_id'].isin(cluster)]
                if len(cluster_roads) > 0:
                    cluster_centroid = cluster_roads.geometry.centroid.unary_union.centroid
                    distance = house_geom.distance(cluster_centroid)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_idx
            
            # Reassign road if house is closer to different cluster
            if best_cluster != house_cluster and min_distance < house_geom.distance(
                roads[roads['road_id'].isin(clusters[house_cluster])].geometry.centroid.unary_union.centroid
            ) * 0.7:  # 30% improvement threshold
                clusters[house_cluster].remove(house_road_id)
                clusters[best_cluster].append(house_road_id)
                road_to_cluster[house_road_id] = best_cluster
        
        return clusters
    

    
    def _create_route_geometry(self, roads: gpd.GeoDataFrame) -> LineString:
        """Create simple route geometry from road segments."""
        if len(roads) == 0:
            return None
        
        try:
            centroids = []
            for _, road in roads.iterrows():
                if hasattr(road, 'geometry') and road.geometry and not road.geometry.is_empty:
                    centroid = road.geometry.centroid
                    if hasattr(centroid, 'x') and hasattr(centroid, 'y'):
                        centroids.append((centroid.x, centroid.y))
            
            if len(centroids) >= 2:
                return LineString(centroids)
            elif len(centroids) == 1:
                x, y = centroids[0]
                return LineString([(x, y), (x + 1, y + 1)])
            else:
                # Create simple line from road bounds
                bounds = roads.total_bounds
                return LineString([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
                
        except Exception as e:
            logger.warning(f"Failed to create route geometry: {e}")
            try:
                bounds = roads.total_bounds
                return LineString([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
            except:
                return LineString([(0, 0), (1, 1)])
    
    def _validate_complete_coverage(self, routes: List[RouteResult], 
                                  active_roads: gpd.GeoDataFrame,
                                  snapped_houses: gpd.GeoDataFrame):
        """Validate complete coverage and no overlaps."""
        # Check for overlapping road segments
        all_segments = set()
        overlaps = set()
        
        for route in routes:
            route_segments = set(route.road_segment_ids)
            overlap = all_segments.intersection(route_segments)
            if overlap:
                overlaps.update(overlap)
            all_segments.update(route_segments)
        
        # Check coverage of all roads with houses
        expected_roads = set(str(rid) for rid in active_roads['road_id'])
        covered_roads = all_segments
        missing_roads = expected_roads - covered_roads
        
        # Check coverage of all houses
        total_houses = len(snapped_houses)
        covered_houses = sum(len(route.ordered_house_ids) for route in routes)
        
        # Log results
        if overlaps:
            logger.error(f"❌ {len(overlaps)} overlapping road segments: {list(overlaps)[:5]}...")
        else:
            logger.info("✅ No overlapping road segments")
            
        if missing_roads:
            logger.error(f"❌ {len(missing_roads)} roads not covered: {list(missing_roads)[:5]}...")
        else:
            logger.info("✅ All roads with houses are covered")
            
        if covered_houses != total_houses:
            logger.error(f"❌ House coverage mismatch: {covered_houses}/{total_houses}")
        else:
            logger.info(f"✅ All {total_houses} houses are covered")
            
        # Summary
        logger.info(f"Route assignment summary: {len(routes)} routes, {len(covered_roads)} roads, {covered_houses} houses")
    
    def reassign_vehicle_routes(self, upload_id: str, unavailable_vehicle_id: str) -> List[RouteResult]:
        """Reassign routes when a vehicle becomes unavailable."""
        logger.info(f"Reassigning routes after vehicle {unavailable_vehicle_id} became unavailable")
        
        # Mark vehicle as unavailable
        blackboard.mark_vehicle_unavailable(upload_id, unavailable_vehicle_id)
        
        # Get current routes and upload data
        current_routes = blackboard.get_routes(upload_id)
        upload_data = blackboard.get_upload_data(upload_id)
        
        if not current_routes or not upload_data:
            raise ValueError("No routes or upload data found for reassignment")
        
        # Find route of unavailable vehicle
        unavailable_route = None
        remaining_routes = []
        
        for route in current_routes:
            if route.vehicle_id == unavailable_vehicle_id:
                unavailable_route = route
            else:
                remaining_routes.append(route)
        
        if not unavailable_route:
            logger.warning(f"No route found for unavailable vehicle {unavailable_vehicle_id}")
            return current_routes
        
        # Redistribute road segments and houses from unavailable vehicle
        roads_to_reassign = unavailable_route.road_segment_ids
        houses_to_reassign = unavailable_route.ordered_house_ids
        
        if not roads_to_reassign or not remaining_routes:
            return remaining_routes
        
        # Distribute road segments evenly among remaining vehicles
        roads_per_vehicle = len(roads_to_reassign) // len(remaining_routes)
        extra_roads = len(roads_to_reassign) % len(remaining_routes)
        
        road_idx = 0
        for i, route in enumerate(remaining_routes):
            roads_to_add = roads_per_vehicle + (1 if i < extra_roads else 0)
            new_roads = roads_to_reassign[road_idx:road_idx + roads_to_add]
            route.road_segment_ids.extend(new_roads)
            road_idx += roads_to_add
        
        # Redistribute houses based on their road assignments
        upload_data = blackboard.get_upload_data(upload_id)
        if upload_data and upload_data.snapped_houses is not None:
            snapped_houses = upload_data.snapped_houses
            
            for route in remaining_routes:
                # Get houses on this route's roads
                route_houses = snapped_houses[snapped_houses['road_id'].isin([int(rid) for rid in route.road_segment_ids])]
                route.ordered_house_ids = [str(h.get('house_id', idx)) for idx, h in route_houses.iterrows()]
        
        # Store updated routes
        blackboard.store_routes(upload_id, remaining_routes)
        
        logger.info(f"Reassigned {len(roads_to_reassign)} road segments and houses to {len(remaining_routes)} vehicles")
        return remaining_routes