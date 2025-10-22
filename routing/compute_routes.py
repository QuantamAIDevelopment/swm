"""Compute shortest paths on road network graph."""
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
from loguru import logger
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

class RouteComputer:
    def __init__(self, road_graph: nx.Graph):
        self.road_graph = road_graph
        
    def compute_cluster_routes(self, clustered_buildings: gpd.GeoDataFrame, depot_location: tuple = None) -> dict:
        """Compute routes for multiple trips with no overlap between trips."""
        routes = {}
        cluster_ids = sorted(clustered_buildings['cluster'].unique())
        
        # Group clusters by trip number to ensure no overlap
        trip_groups = {}
        for cluster_id in cluster_ids:
            if 'trip_' in str(cluster_id):
                trip_num = int(str(cluster_id).split('_')[1])
                if trip_num not in trip_groups:
                    trip_groups[trip_num] = []
                trip_groups[trip_num].append(cluster_id)
            else:
                # Fallback for non-trip clusters
                if 'default' not in trip_groups:
                    trip_groups['default'] = []
                trip_groups['default'].append(cluster_id)
        
        # Process each trip separately to ensure no overlap
        for trip_num, trip_cluster_ids in trip_groups.items():
            logger.info(f"Processing Trip {trip_num} with {len(trip_cluster_ids)} vehicle routes")
            previous_end = depot_location
            
            for cluster_id in trip_cluster_ids:
                cluster_buildings = clustered_buildings[clustered_buildings['cluster'] == cluster_id]
                
                if len(cluster_buildings) == 0:
                    continue
                
                # Use previous vehicle's end as this vehicle's start within the same trip
                if previous_end is None:
                    start_point = cluster_buildings.iloc[0]['road_node']
                else:
                    start_point = previous_end
                
                route = self._solve_vrp_for_cluster(cluster_buildings, start_point)
                route['trip_number'] = trip_num if trip_num != 'default' else 1
                route['houses_count'] = len(cluster_buildings)
                routes[cluster_id] = route
                
                # Set end point for next vehicle in the same trip
                if route['nodes']:
                    previous_end = route['nodes'][-1]
        
        logger.info(f"Computed {len(routes)} routes across {len(trip_groups)} trips")
        return routes
    
    def _solve_vrp_for_cluster(self, buildings: gpd.GeoDataFrame, depot: tuple) -> dict:
        """Solve VRP for a single cluster using OR-Tools."""
        locations = [depot] + [building['road_node'] for _, building in buildings.iterrows()]
        
        # Create distance matrix using NetworkX shortest paths
        distance_matrix = self._create_distance_matrix(locations)
        
        # Create VRP model
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convert to int
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_solution(manager, routing, solution, locations)
        else:
            logger.warning("No solution found for cluster, using simple path")
            return self._create_simple_path(locations)
    
    def _create_distance_matrix(self, locations: list) -> np.ndarray:
        """Create distance matrix using NetworkX shortest paths."""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(
                            self.road_graph, 
                            locations[i], 
                            locations[j], 
                            weight='weight'
                        )
                        matrix[i][j] = path_length
                    except nx.NetworkXNoPath:
                        # Use Euclidean distance as fallback
                        p1, p2 = Point(locations[i]), Point(locations[j])
                        matrix[i][j] = p1.distance(p2) * 10  # Penalty for no path
                        
        return matrix
    
    def _extract_solution(self, manager, routing, solution, locations: list) -> dict:
        """Extract route solution from OR-Tools."""
        route_nodes = []
        route_paths = []
        total_distance = 0
        
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_nodes.append(locations[node_index])
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            if not routing.IsEnd(index):
                next_node_index = manager.IndexToNode(index)
                # Get actual path between nodes
                try:
                    path = nx.shortest_path(
                        self.road_graph,
                        locations[node_index],
                        locations[next_node_index],
                        weight='weight'
                    )
                    path_geom = self._path_to_linestring(path)
                    route_paths.append(path_geom)
                    
                    path_length = nx.shortest_path_length(
                        self.road_graph,
                        locations[node_index],
                        locations[next_node_index],
                        weight='weight'
                    )
                    total_distance += path_length
                except nx.NetworkXNoPath:
                    # Direct line as fallback
                    direct_line = LineString([locations[node_index], locations[next_node_index]])
                    route_paths.append(direct_line)
                    total_distance += direct_line.length
        
        # Create overall geometry from all path coordinates
        all_coords = []
        for path in route_paths:
            all_coords.extend(path.coords)
        
        # Ensure we have at least 2 points for LineString
        if len(all_coords) < 2:
            if route_nodes:
                all_coords = route_nodes[:2] if len(route_nodes) >= 2 else [route_nodes[0], route_nodes[0]]
            else:
                all_coords = [(0, 0), (0, 0)]
        
        # Ensure valid geometry
        if len(all_coords) < 2:
            all_coords = route_nodes[:2] if len(route_nodes) >= 2 else [(0, 0), (0, 0)]
        
        return {
            'nodes': route_nodes,
            'paths': route_paths,
            'total_distance': total_distance,
            'geometry': LineString(all_coords)
        }
    
    def _create_simple_path(self, locations: list) -> dict:
        """Create simple path when VRP fails."""
        route_paths = []
        total_distance = 0
        
        for i in range(len(locations) - 1):
            try:
                path = nx.shortest_path(
                    self.road_graph,
                    locations[i],
                    locations[i + 1],
                    weight='weight'
                )
                path_geom = self._path_to_linestring(path)
                route_paths.append(path_geom)
                
                path_length = nx.shortest_path_length(
                    self.road_graph,
                    locations[i],
                    locations[i + 1],
                    weight='weight'
                )
                total_distance += path_length
            except nx.NetworkXNoPath:
                direct_line = LineString([locations[i], locations[i + 1]])
                route_paths.append(direct_line)
                total_distance += direct_line.length
        
        # Create overall geometry from all path coordinates
        all_coords = []
        for path in route_paths:
            all_coords.extend(path.coords)
        
        # Ensure we have at least 2 points for LineString
        if len(all_coords) < 2:
            if locations:
                all_coords = locations[:2] if len(locations) >= 2 else [locations[0], locations[0]]
            else:
                all_coords = [(0, 0), (0, 0)]
        
        # Ensure valid geometry
        if len(all_coords) < 2:
            all_coords = locations[:2] if len(locations) >= 2 else [(0, 0), (0, 0)]
        
        return {
            'nodes': locations,
            'paths': route_paths,
            'total_distance': total_distance,
            'geometry': LineString(all_coords)
        }
    
    def _path_to_linestring(self, path: list) -> LineString:
        """Convert node path to LineString geometry."""
        if len(path) < 2:
            # Return a minimal line for single points
            if len(path) == 1:
                point = path[0]
                return LineString([point, point])
            else:
                return LineString([(0, 0), (0, 0)])
        return LineString(path)