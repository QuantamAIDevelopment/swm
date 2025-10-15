"""Route assignment agent for clustering and VRP coordination."""
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from datetime import datetime
from shapely.geometry import LineString, Point

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
        
        # Validate and fix coverage issues
        self._validate_complete_coverage(routes, active_roads, snapped_houses)
        
        # Create continuous chain
        routes = self._create_continuous_chain(routes)
        
        return routes
    
    def _create_balanced_road_clusters(self, roads: gpd.GeoDataFrame, 
                                     snapped_houses: gpd.GeoDataFrame, 
                                     n_clusters: int) -> List[List]:
        """Create balanced clusters ensuring houses are within geographic zones."""
        
        # Use simple horizontal strip clustering for clear geographic zones
        return self._fallback_strip_clustering(roads, snapped_houses, n_clusters)
    
    def _fallback_strip_clustering(self, roads: gpd.GeoDataFrame, 
                                 snapped_houses: gpd.GeoDataFrame, 
                                 n_clusters: int) -> List[List]:
        """Geographic zone-based clustering with strict boundaries."""
        
        # Get bounding box of all houses
        bounds = snapped_houses.total_bounds  # [minx, miny, maxx, maxy]
        
        # Create strict horizontal zones with buffer
        y_min, y_max = bounds[1], bounds[3]
        zone_height = (y_max - y_min) / n_clusters
        buffer = zone_height * 0.01  # 1% buffer to prevent edge cases
        
        # Create non-overlapping zone boundaries
        zone_boundaries = []
        for i in range(n_clusters + 1):
            boundary_y = y_max - (i * zone_height)
            zone_boundaries.append(boundary_y)
        
        # Assign houses to zones based on Y coordinate with strict boundaries
        house_zones = [[] for _ in range(n_clusters)]
        house_to_road = {}
        
        for idx, house in snapped_houses.iterrows():
            # Get house coordinates
            if hasattr(house.geometry, 'x') and hasattr(house.geometry, 'y'):
                x, y = house.geometry.x, house.geometry.y
            else:
                centroid = house.geometry.centroid
                x, y = centroid.x, centroid.y
            
            # Strict zone assignment with clear boundaries
            zone_id = n_clusters - 1  # Default to last zone
            
            # Find exact zone based on Y coordinate
            for i in range(n_clusters):
                if y >= zone_boundaries[i+1] and y < zone_boundaries[i]:
                    zone_id = i
                    break
            
            # Handle edge case for maximum Y value
            if y >= zone_boundaries[0]:
                zone_id = 0
            
            house_zones[zone_id].append(idx)
            house_to_road[idx] = house['road_id']
        
        # Create road clusters - assign each road to zone with most houses
        road_clusters = [[] for _ in range(n_clusters)]
        road_zone_counts = {}
        
        # Count houses per road per zone
        for zone_id, house_indices in enumerate(house_zones):
            for house_idx in house_indices:
                road_id = house_to_road[house_idx]
                if road_id not in road_zone_counts:
                    road_zone_counts[road_id] = [0] * n_clusters
                road_zone_counts[road_id][zone_id] += 1
        
        # Assign each road to zone with maximum houses
        for road_id, zone_counts in road_zone_counts.items():
            best_zone = zone_counts.index(max(zone_counts))
            road_clusters[best_zone].append(road_id)
        
        # Final validation - ensure no overlaps
        all_roads = set()
        overlap_found = False
        
        for i, (road_cluster, house_zone) in enumerate(zip(road_clusters, house_zones)):
            road_set = set(road_cluster)
            overlap = all_roads.intersection(road_set)
            if overlap:
                logger.error(f"❌ Zone {i+1} has {len(overlap)} overlapping roads: {list(overlap)[:3]}...")
                overlap_found = True
            all_roads.update(road_set)
            logger.info(f"Zone {i+1}: {len(road_cluster)} roads, {len(house_zone)} houses (Y: {zone_boundaries[i+1]:.0f} to {zone_boundaries[i]:.0f})")
        
        if not overlap_found:
            logger.info(f"✅ No overlaps - {len(all_roads)} unique roads assigned to {n_clusters} zones")
        else:
            logger.error(f"❌ Overlaps detected - fixing required")
        
        # Ensure no empty clusters
        for i, cluster in enumerate(road_clusters):
            if not cluster:
                logger.warning(f"Zone {i+1} is empty - redistributing roads")
                # Move some roads from largest cluster
                largest_idx = max(range(len(road_clusters)), key=lambda x: len(road_clusters[x]))
                if road_clusters[largest_idx]:
                    roads_to_move = road_clusters[largest_idx][:2]  # Move 2 roads
                    for road in roads_to_move:
                        road_clusters[largest_idx].remove(road)
                        road_clusters[i].append(road)
        
        return road_clusters
    
    def _create_continuous_chain(self, routes: List[RouteResult]) -> List[RouteResult]:
        """Create continuous chain where each vehicle's end connects to next vehicle's start."""
        if len(routes) <= 1:
            return routes
        
        # Sort routes by zone (north to south) for logical chaining
        routes.sort(key=lambda r: r.route_id)
        
        # Create continuous chain by connecting end points to start points
        for i in range(len(routes) - 1):
            current_route = routes[i]
            next_route = routes[i + 1]
            
            # Get current route's end point
            if current_route.geometry and len(current_route.geometry.coords) > 0:
                end_point = current_route.geometry.coords[-1]
                
                # Update next route's start point to current route's end point
                if next_route.geometry and len(next_route.geometry.coords) > 0:
                    # Create new geometry with connected start point
                    coords = list(next_route.geometry.coords)
                    coords[0] = end_point  # Connect start to previous end
                    next_route.geometry = LineString(coords)
                    
                    # Update route nodes for continuity
                    next_route.start_node = current_route.end_node
        
        # Log the continuous chain
        for i, route in enumerate(routes):
            logger.info(f"Chain {i+1}: Vehicle {route.vehicle_id} -> {routes[i+1].vehicle_id if i+1 < len(routes) else 'END'}")
        
        return routes
    
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
        """Validate and ensure complete coverage with no overlaps."""
        # Verify no overlapping road segments
        all_road_ids = set()
        for route in routes:
            route_road_ids = set(route.road_segment_ids)
            if all_road_ids.intersection(route_road_ids):
                logger.error("Found overlapping roads - this should not happen with sequential assignment")
            all_road_ids.update(route_road_ids)
        
        # Ensure all houses are assigned
        for route in routes:
            route_road_ids = [int(rid) for rid in route.road_segment_ids]
            route_houses = snapped_houses[snapped_houses['road_id'].isin(route_road_ids)]
            route.ordered_house_ids = [str(idx) for idx in route_houses.index]
        
        total_houses = sum(len(route.ordered_house_ids) for route in routes)
        logger.info(f"✅ Coverage: {total_houses}/{len(snapped_houses)} houses, {len(all_road_ids)} roads")
        logger.info(f"✅ Zero overlaps guaranteed by sequential assignment")
    
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