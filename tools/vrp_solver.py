"""VRP solver using OR-Tools for route optimization."""
import logging
import numpy as np
import networkx as nx
import geopandas as gpd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Dict, Tuple, Optional
from shapely.geometry import LineString, Point
from configurations.config import Config

logger = logging.getLogger(__name__)

class VRPSolver:
    def __init__(self, road_network: gpd.GeoDataFrame):
        self.road_network = road_network
        self.graph = self._build_road_graph()
    
    def _build_road_graph(self) -> nx.Graph:
        """Build NetworkX graph from road network."""
        G = nx.Graph()
        
        for idx, road in self.road_network.iterrows():
            road_id = road.get('road_id', idx)
            geom = road.geometry
            
            # Handle different geometry types
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                self._add_road_to_graph(G, road_id, coords, geom)
            elif geom.geom_type == 'MultiLineString':
                # Handle MultiLineString by processing each part
                for i, line in enumerate(geom.geoms):
                    coords = list(line.coords)
                    part_id = f"{road_id}_{i}"
                    self._add_road_to_graph(G, part_id, coords, line)
            else:
                logger.warning(f"Unsupported geometry type: {geom.geom_type} for road {road_id}")
        
        logger.info(f"Built road graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _add_road_to_graph(self, G, road_id, coords, geometry):
        """Add a road segment to the graph."""
        start_node = f"road_{road_id}_start"
        end_node = f"road_{road_id}_end"
        
        G.add_node(start_node, pos=coords[0], road_id=road_id)
        G.add_node(end_node, pos=coords[-1], road_id=road_id)
        
        # Add edge with distance as weight
        distance = geometry.length
        G.add_edge(start_node, end_node, weight=distance, road_id=road_id, geometry=geometry)
    
    def _calculate_distance_matrix(self, nodes: gpd.GeoDataFrame, use_osrm: bool = True) -> np.ndarray:
        """Calculate distance matrix using OSRM or road network."""
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        if use_osrm:
            # Use OSRM for more accurate real-world distances
            try:
                from tools.osrm_routing import OSRMRouter
                osrm = OSRMRouter()
                
                locations = [(node.geometry.x, node.geometry.y) for _, node in nodes.iterrows()]
                distance_matrix = osrm.get_distance_matrix(locations)
                
                logger.info("Using OSRM distance matrix")
                return distance_matrix
                
            except Exception as e:
                logger.warning(f"OSRM failed, falling back to road network: {e}")
        
        # Fallback to road network calculation
        temp_nodes = []
        for idx, node in nodes.iterrows():
            node_id = f"temp_{idx}"
            temp_nodes.append(node_id)
            self.graph.add_node(node_id, pos=(node.geometry.x, node.geometry.y))
            
            # Connect to nearest road nodes
            min_dist = float('inf')
            nearest_road_node = None
            for road_node in self.graph.nodes():
                if road_node.startswith('road_'):
                    road_pos = self.graph.nodes[road_node]['pos']
                    dist = ((node.geometry.x - road_pos[0])**2 + (node.geometry.y - road_pos[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_road_node = road_node
            
            if nearest_road_node:
                self.graph.add_edge(node_id, nearest_road_node, weight=min_dist)
        
        # Calculate shortest paths
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    try:
                        path_length = nx.shortest_path_length(
                            self.graph, temp_nodes[i], temp_nodes[j], weight='weight'
                        )
                        distance_matrix[i][j] = path_length
                    except nx.NetworkXNoPath:
                        # Use Haversine distance as fallback
                        node_i = nodes.iloc[i]
                        node_j = nodes.iloc[j]
                        distance_matrix[i][j] = self._haversine_distance(
                            (node_i.geometry.x, node_i.geometry.y),
                            (node_j.geometry.x, node_j.geometry.y)
                        )
        
        # Remove temporary nodes
        for temp_node in temp_nodes:
            if temp_node in self.graph:
                self.graph.remove_node(temp_node)
        
        return distance_matrix
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate Haversine distance between two coordinates."""
        R = 6371000  # Earth radius in meters
        lat1, lon1 = np.radians(coord1[1]), np.radians(coord1[0])
        lat2, lon2 = np.radians(coord2[1]), np.radians(coord2[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def solve_vrp(self, cluster_nodes: gpd.GeoDataFrame, vehicle_start_idx: int = 0) -> Optional[List[int]]:
        """
        Solve VRP for a cluster of nodes.
        
        Args:
            cluster_nodes: GeoDataFrame of nodes to visit
            vehicle_start_idx: Index of starting node
            
        Returns:
            Ordered list of node indices representing the route
        """
        if len(cluster_nodes) <= 1:
            return list(range(len(cluster_nodes)))
        
        logger.info(f"Solving VRP for {len(cluster_nodes)} nodes")
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(cluster_nodes)
        
        # Convert to integer (OR-Tools requirement)
        distance_matrix = (distance_matrix * 1000).astype(int)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(cluster_nodes), 1, vehicle_start_idx
        )
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = Config.VRP_TIME_LIMIT_SECONDS
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))  # Add end node
            
            total_distance = solution.ObjectiveValue() / 1000.0  # Convert back to original units
            logger.info(f"VRP solution found with total distance: {total_distance:.2f}")
            return route
        else:
            logger.warning("No VRP solution found")
            return []
    
    def build_route_geometry(self, ordered_nodes: List[int], cluster_nodes: gpd.GeoDataFrame) -> LineString:
        """Build route geometry following road network."""
        if len(ordered_nodes) < 2:
            return LineString([cluster_nodes.iloc[0].geometry.coords[0]] * 2)
        
        route_coords = []
        for i in range(len(ordered_nodes) - 1):
            start_node = cluster_nodes.iloc[ordered_nodes[i]]
            end_node = cluster_nodes.iloc[ordered_nodes[i + 1]]
            
            # Add start point
            if i == 0:
                route_coords.append((start_node.geometry.x, start_node.geometry.y))
            
            # Add end point
            route_coords.append((end_node.geometry.x, end_node.geometry.y))
        
        return LineString(route_coords)