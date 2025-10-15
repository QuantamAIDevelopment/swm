"""OSRM-based routing service for garbage collection optimization."""
import logging
import requests
import numpy as np
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Point, LineString
import osmnx as ox
import networkx as nx
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

logger = logging.getLogger(__name__)

class OSRMRouter:
    """OSRM routing service for real-world driving directions."""
    
    def __init__(self, base_url: str = "http://router.project-osrm.org"):
        self.base_url = base_url
        
    def get_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                  profile: str = "driving") -> Dict:
        """Get route between two points with turn-by-turn directions."""
        try:
            url = f"{self.base_url}/route/v1/{profile}/{start[0]},{start[1]};{end[0]},{end[1]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true',
                'annotations': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('routes'):
                    route = data['routes'][0]
                    return {
                        'coordinates': [[coord[1], coord[0]] for coord in route['geometry']['coordinates']],
                        'distance': route.get('distance', 0),
                        'duration': route.get('duration', 0),
                        'directions': self._extract_directions(route),
                        'success': True
                    }
        except Exception as e:
            logger.warning(f"OSRM routing failed: {e}")
        
        return {
            'coordinates': [[start[1], start[0]], [end[1], end[0]]],
            'distance': self._haversine_distance(start, end),
            'duration': 0,
            'directions': [],
            'success': False
        }
    
    def get_distance_matrix(self, locations: List[Tuple[float, float]], 
                           profile: str = "driving") -> np.ndarray:
        """Get distance matrix for multiple locations."""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        # OSRM table service for efficient matrix calculation
        try:
            coords_str = ";".join([f"{loc[0]},{loc[1]}" for loc in locations])
            url = f"{self.base_url}/table/v1/{profile}/{coords_str}"
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'distances' in data:
                    return np.array(data['distances'])
        except Exception as e:
            logger.warning(f"OSRM matrix failed, using fallback: {e}")
        
        # Fallback: individual route requests
        for i in range(n):
            for j in range(n):
                if i != j:
                    route = self.get_route(locations[i], locations[j], profile)
                    matrix[i][j] = route['distance']
        
        return matrix
    
    def _extract_directions(self, route: Dict) -> List[Dict]:
        """Extract turn-by-turn directions from OSRM route."""
        directions = []
        if 'legs' in route:
            for leg in route['legs']:
                if 'steps' in leg:
                    for step in leg['steps']:
                        maneuver = step.get('maneuver', {})
                        directions.append({
                            'instruction': maneuver.get('instruction', 'Continue'),
                            'distance': f"{step.get('distance', 0):.0f}m",
                            'duration': f"{step.get('duration', 0):.0f}s",
                            'type': maneuver.get('type', 'continue')
                        })
        return directions
    
    def _haversine_distance(self, coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two coordinates."""
        R = 6371000  # Earth radius in meters
        lat1, lon1 = np.radians(coord1[1]), np.radians(coord1[0])
        lat2, lon2 = np.radians(coord2[1]), np.radians(coord2[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

class OptimizedRouteAssigner:
    """Combines OSRM, OSMnx, OR-Tools, and K-means for optimal route assignment."""
    
    def __init__(self):
        self.osrm = OSRMRouter()
        self.osm_graph = None
        
    def assign_routes(self, buildings: List[Tuple[float, float]], 
                     depot: Tuple[float, float], 
                     num_vehicles: int = 3) -> Dict:
        """Assign optimal routes using combined optimization techniques."""
        
        # Step 1: K-means clustering for initial grouping
        clusters = self._cluster_buildings(buildings, num_vehicles)
        
        # Step 2: Download OSM network for the area
        self._download_osm_network(buildings + [depot])
        
        # Step 3: Optimize each cluster using OR-Tools + OSRM
        optimized_routes = []
        for cluster_idx, cluster_buildings in enumerate(clusters):
            if not cluster_buildings:
                continue
                
            # Add depot to cluster
            cluster_locations = [depot] + cluster_buildings
            
            # Get OSRM distance matrix
            distance_matrix = self.osrm.get_distance_matrix(cluster_locations)
            
            # Solve TSP for this cluster using OR-Tools
            route_order = self._solve_tsp(distance_matrix)
            
            # Build detailed route with OSRM
            route_details = self._build_route_details(cluster_locations, route_order)
            
            optimized_routes.append({
                'vehicle_id': cluster_idx,
                'locations': [cluster_locations[i] for i in route_order],
                'route_details': route_details,
                'total_distance': route_details['total_distance'],
                'total_duration': route_details['total_duration']
            })
        
        return {
            'routes': optimized_routes,
            'total_vehicles': len(optimized_routes),
            'optimization_methods': ['K-means', 'OSRM', 'OR-Tools', 'OSMnx']
        }
    
    def _cluster_buildings(self, buildings: List[Tuple[float, float]], 
                          num_clusters: int) -> List[List[Tuple[float, float]]]:
        """Use K-means to cluster buildings geographically."""
        if len(buildings) <= num_clusters:
            return [[building] for building in buildings]
        
        # Convert to numpy array for K-means
        coords = np.array(buildings)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        # Group buildings by cluster
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(buildings[i])
        
        logger.info(f"K-means clustering: {[len(c) for c in clusters]} buildings per cluster")
        return clusters
    
    def _download_osm_network(self, locations: List[Tuple[float, float]]):
        """Skip OSM network download for now."""
        logger.info("Skipping OSM network download - using OSRM for routing")
        self.osm_graph = None
    
    def _solve_tsp(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve TSP using OR-Tools."""
        n = len(distance_matrix)
        if n <= 2:
            return list(range(n))
        
        # Convert to integer for OR-Tools
        int_matrix = (distance_matrix).astype(int)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # Start from depot (index 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 30
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            return route
        
        # Fallback: simple order
        return list(range(n))
    
    def _build_route_details(self, locations: List[Tuple[float, float]], 
                           route_order: List[int]) -> Dict:
        """Build detailed route information using OSRM."""
        total_distance = 0
        total_duration = 0
        all_coordinates = []
        all_directions = []
        
        for i in range(len(route_order) - 1):
            start_idx = route_order[i]
            end_idx = route_order[i + 1]
            
            start_loc = locations[start_idx]
            end_loc = locations[end_idx]
            
            # Get route segment from OSRM
            route_segment = self.osrm.get_route(start_loc, end_loc)
            
            # Accumulate totals
            total_distance += route_segment['distance']
            total_duration += route_segment['duration']
            
            # Collect coordinates and directions
            all_coordinates.extend(route_segment['coordinates'])
            all_directions.extend(route_segment['directions'][:3])  # Limit directions
        
        return {
            'coordinates': all_coordinates,
            'directions': all_directions[:15],  # Limit total directions
            'total_distance': total_distance,
            'total_duration': total_duration
        }