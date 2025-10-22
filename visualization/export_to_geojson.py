"""Export routes to GeoJSON and summary CSV."""
import geopandas as gpd
import pandas as pd
import json
import os
import tempfile
from loguru import logger
from shapely.geometry import LineString
from typing import Dict, List

class RouteExporter:
    def __init__(self, export_dir: str = None):
        self.routes_gdf = None
        self.summary_df = None
        # Set export directory outside codebase
        self.export_dir = export_dir or os.path.join(tempfile.gettempdir(), 'swm_exports')
        os.makedirs(self.export_dir, exist_ok=True)
    
    def prepare_routes_geojson(self, routes: Dict, vehicles_df: pd.DataFrame, directions: Dict = None) -> gpd.GeoDataFrame:
        """Prepare routes data for GeoJSON export."""
        route_features = []
        
        for cluster_id, route_data in routes.items():
            # Get vehicle info for this cluster
            vehicle_info = vehicles_df.iloc[cluster_id] if cluster_id < len(vehicles_df) else {}
            
            # Get directions for this route if available
            route_directions = directions.get(cluster_id, {}) if directions else {}
            
            feature = {
                'cluster_id': cluster_id,
                'vehicle_id': vehicle_info.get('vehicle_id', f'vehicle_{cluster_id}'),
                'vehicle_type': vehicle_info.get('type', 'standard'),
                'capacity': vehicle_info.get('capacity', 1000),
                'total_distance': route_data['total_distance'],
                'total_duration': route_directions.get('total_duration', 0),
                'num_stops': len(route_data['nodes']) - 1,  # Exclude depot
                'start_point': route_data['nodes'][0] if route_data['nodes'] else None,
                'end_point': route_data['nodes'][-1] if route_data['nodes'] else None,
                'geometry': route_data['geometry']
            }
            
            # Add direction steps as properties
            if route_directions.get('steps'):
                feature['directions'] = self._format_directions(route_directions['steps'])
                feature['color_coded_steps'] = len([s for s in route_directions['steps'] if s.get('color')])
            
            route_features.append(feature)
        
        self.routes_gdf = gpd.GeoDataFrame(route_features)
        logger.info(f"Prepared {len(route_features)} routes for export")
        return self.routes_gdf
    
    def export_routes_geojson(self, filename: str = "routes.geojson") -> str:
        """Export routes to GeoJSON file outside codebase."""
        if self.routes_gdf is None:
            raise ValueError("No routes data prepared")
        
        # Export to temp directory, not codebase
        output_path = os.path.join(self.export_dir, filename)
        
        # Clean data - remove any response fields
        clean_gdf = self.routes_gdf.copy()
        exclude_fields = ['html_response', 'json_response', 'api_response', 'raw_output']
        for field in exclude_fields:
            if field in clean_gdf.columns:
                clean_gdf = clean_gdf.drop(columns=[field])
        
        if clean_gdf.crs is None:
            clean_gdf.set_crs(epsg=4326, inplace=True)
        
        clean_gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Exported to temp location: {output_path}")
        return output_path
    
    def prepare_summary_csv(self, routes: Dict, vehicles_df: pd.DataFrame, directions: Dict = None) -> pd.DataFrame:
        """Prepare summary statistics for CSV export."""
        summary_data = []
        total_distance = 0
        total_duration = 0
        
        for cluster_id, route_data in routes.items():
            vehicle_info = vehicles_df.iloc[cluster_id] if cluster_id < len(vehicles_df) else {}
            route_directions = directions.get(cluster_id, {}) if directions else {}
            
            route_distance = route_data['total_distance']
            route_duration = route_directions.get('total_duration', 0)
            
            summary_row = {
                'vehicle_id': vehicle_info.get('vehicle_id', f'vehicle_{cluster_id}'),
                'cluster_id': cluster_id,
                'vehicle_type': vehicle_info.get('type', 'standard'),
                'capacity_kg': vehicle_info.get('capacity', 1000),
                'num_stops': len(route_data['nodes']) - 1,
                'route_distance_m': round(route_distance, 2),
                'route_duration_s': round(route_duration, 2),
                'route_duration_min': round(route_duration / 60, 2),
                'avg_distance_per_stop': round(route_distance / max(1, len(route_data['nodes']) - 1), 2),
                'efficiency_score': self._calculate_efficiency_score(route_data, vehicle_info),
                'start_lat': route_data['nodes'][0][1] if route_data['nodes'] else None,
                'start_lon': route_data['nodes'][0][0] if route_data['nodes'] else None,
                'end_lat': route_data['nodes'][-1][1] if route_data['nodes'] else None,
                'end_lon': route_data['nodes'][-1][0] if route_data['nodes'] else None
            }
            
            # Add direction statistics
            if route_directions.get('steps'):
                steps = route_directions['steps']
                summary_row.update({
                    'num_turns': len([s for s in steps if 'turn' in s.get('maneuver_type', '')]),
                    'num_straight': len([s for s in steps if s.get('color') == 'green']),
                    'num_complex_maneuvers': len([s for s in steps if s.get('color') == 'red'])
                })
            
            summary_data.append(summary_row)
            total_distance += route_distance
            total_duration += route_duration
        
        # Add summary row
        summary_data.append({
            'vehicle_id': 'TOTAL',
            'cluster_id': -1,
            'vehicle_type': 'summary',
            'capacity_kg': vehicles_df['capacity'].sum() if 'capacity' in vehicles_df.columns else 0,
            'num_stops': sum(len(route['nodes']) - 1 for route in routes.values()),
            'route_distance_m': round(total_distance, 2),
            'route_duration_s': round(total_duration, 2),
            'route_duration_min': round(total_duration / 60, 2),
            'avg_distance_per_stop': round(total_distance / max(1, sum(len(route['nodes']) - 1 for route in routes.values())), 2),
            'efficiency_score': round(total_distance / len(routes), 2) if routes else 0
        })
        
        self.summary_df = pd.DataFrame(summary_data)
        logger.info(f"Prepared summary for {len(routes)} routes")
        return self.summary_df
    
    def export_summary_csv(self, filename: str = "summary.csv") -> str:
        """Export summary to CSV file outside codebase."""
        if self.summary_df is None:
            raise ValueError("No summary data prepared")
        
        # Export to temp directory, not codebase
        output_path = os.path.join(self.export_dir, filename)
        
        # Clean data - remove any response fields
        clean_df = self.summary_df.copy()
        exclude_fields = ['html_response', 'json_response', 'api_response', 'raw_output']
        for field in exclude_fields:
            if field in clean_df.columns:
                clean_df = clean_df.drop(columns=[field])
        
        clean_df.to_csv(output_path, index=False)
        logger.info(f"Exported to temp location: {output_path}")
        return output_path
    
    def _format_directions(self, steps: List[Dict]) -> List[Dict]:
        """Format direction steps for export."""
        formatted_steps = []
        for step in steps:
            formatted_step = {
                'instruction': step.get('instruction', ''),
                'type': step.get('maneuver_type', ''),
                'modifier': step.get('modifier', ''),
                'color': step.get('color', 'green'),
                'emoji': step.get('emoji', '🟢'),
                'distance_m': round(step.get('distance', 0), 2),
                'duration_s': round(step.get('duration', 0), 2)
            }
            formatted_steps.append(formatted_step)
        return formatted_steps
    
    def _calculate_efficiency_score(self, route_data: Dict, vehicle_info: Dict) -> float:
        """Calculate efficiency score for the route."""
        distance = route_data['total_distance']
        num_stops = len(route_data['nodes']) - 1
        capacity = vehicle_info.get('capacity', 1000)
        
        if num_stops == 0:
            return 0.0
        
        # Simple efficiency: distance per stop normalized by capacity
        base_score = distance / num_stops
        capacity_factor = min(1.0, capacity / 1000)  # Normalize to 1000kg baseline
        
        return round(base_score * capacity_factor, 2)