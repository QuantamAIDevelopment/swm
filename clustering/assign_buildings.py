"""Cluster buildings based on number of available vehicles."""
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN
from loguru import logger
import numpy as np
from services.vehicle_service import VehicleService

class BuildingClusterer:
    def __init__(self):
        self.clusters = None
        self.vehicle_service = VehicleService()
        
    def load_vehicles(self, csv_path: str = None) -> pd.DataFrame:
        """Load vehicle data from live API or fallback to CSV."""
        try:
            # Try to get live vehicle data first
            vehicles_df = self.vehicle_service.get_live_vehicles()
            
            if vehicles_df is not None and len(vehicles_df) > 0:
                logger.info(f"Loaded {len(vehicles_df)} vehicles from live API")
                return vehicles_df
            
            # Fallback to CSV if provided
            if csv_path:
                logger.warning("Live API failed, falling back to CSV")
                vehicles_df = pd.read_csv(csv_path)
                active_vehicles = vehicles_df[vehicles_df.get('status', 'active') == 'active']
                logger.info(f"Loaded {len(active_vehicles)} active vehicles from {csv_path}")
                return active_vehicles
            
            # Create fallback data if no CSV provided
            logger.warning("No CSV provided, using fallback vehicle data")
            return self.vehicle_service._create_fallback_vehicles()
            
        except Exception as e:
            logger.error(f"Failed to load vehicles: {e}")
            # Return fallback data instead of raising
            return self.vehicle_service._create_fallback_vehicles()
    
    def cluster_buildings(self, buildings_gdf: gpd.GeoDataFrame, num_vehicles: int, method='kmeans') -> gpd.GeoDataFrame:
        """Cluster buildings based on number of vehicles."""
        if len(buildings_gdf) == 0:
            logger.warning("No buildings to cluster")
            return buildings_gdf
            
        # Extract coordinates
        coords = np.array([[geom.x, geom.y] for geom in buildings_gdf.geometry])
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=num_vehicles, random_state=42, n_init=10)
        elif method == 'dbscan':
            # Auto-adjust eps based on data spread
            eps = self._calculate_optimal_eps(coords)
            clusterer = DBSCAN(eps=eps, min_samples=max(2, len(coords) // (num_vehicles * 3)))
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(coords)
        
        # Handle DBSCAN noise points
        if method == 'dbscan':
            cluster_labels = self._reassign_noise_points(cluster_labels, coords, num_vehicles)
        
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf['cluster'] = cluster_labels
        
        logger.info(f"Clustered {len(buildings_gdf)} buildings into {len(set(cluster_labels))} clusters")
        return buildings_gdf
    
    def _calculate_optimal_eps(self, coords: np.ndarray) -> float:
        """Calculate optimal eps for DBSCAN based on data spread."""
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=4)
        neighbors_fit = neighbors.fit(coords)
        distances, indices = neighbors_fit.kneighbors(coords)
        distances = np.sort(distances[:, 3], axis=0)
        
        # Use knee point detection or simple heuristic
        return np.percentile(distances, 75)
    
    def _reassign_noise_points(self, labels: np.ndarray, coords: np.ndarray, target_clusters: int) -> np.ndarray:
        """Reassign noise points (-1) to nearest valid clusters."""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # If we have fewer clusters than vehicles, use KMeans fallback
        unique_labels = set(labels[~noise_mask])
        if len(unique_labels) < target_clusters:
            kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(coords)
        
        # Assign noise points to nearest cluster centroid
        for i, label in enumerate(labels):
            if label == -1:
                min_dist = float('inf')
                best_cluster = 0
                point = coords[i]
                
                for cluster_id in unique_labels:
                    cluster_points = coords[labels == cluster_id]
                    centroid = np.mean(cluster_points, axis=0)
                    dist = np.linalg.norm(point - centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_id
                
                labels[i] = best_cluster
        
        return labels
    
    def get_cluster_summary(self, buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate cluster summary statistics."""
        summary = buildings_gdf.groupby('cluster').agg({
            'geometry': 'count',
            'snap_distance': ['mean', 'max']
        }).round(4)
        
        summary.columns = ['building_count', 'avg_snap_distance', 'max_snap_distance']
        summary = summary.reset_index()
        
        logger.info(f"Generated summary for {len(summary)} clusters")
        return summary