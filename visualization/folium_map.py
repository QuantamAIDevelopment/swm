"""Create interactive Folium maps with color-coded routes."""
import folium
import geopandas as gpd
import pandas as pd
from loguru import logger
import json
from typing import Dict, List
import random

class FoliumMapGenerator:
    def __init__(self):
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
    
    def create_route_map(self, routes_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame = None, 
                        center_coords: tuple = None, zoom_start: int = 12) -> folium.Map:
        """Create interactive map with color-coded routes."""
        
        # Determine map center
        if center_coords is None:
            bounds = routes_gdf.total_bounds
            center_coords = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create base map
        m = folium.Map(
            location=center_coords,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add route layers
        self._add_route_layers(m, routes_gdf)
        
        # Add building markers if provided
        if buildings_gdf is not None:
            self._add_building_markers(m, buildings_gdf)
        
        # Add legend
        self._add_legend(m, routes_gdf)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        logger.info(f"Created Folium map with {len(routes_gdf)} routes")
        return m
    
    def _add_route_layers(self, map_obj: folium.Map, routes_gdf: gpd.GeoDataFrame):
        """Add route layers to map with different colors per vehicle."""
        
        for idx, route in routes_gdf.iterrows():
            color = self.color_palette[idx % len(self.color_palette)]
            vehicle_id = route.get('vehicle_id', f'Vehicle {idx}')
            
            # Create feature group for this route
            route_group = folium.FeatureGroup(name=f"🚛 {vehicle_id}")
            
            # Add route line
            if route.geometry is not None:
                route_coords = [[lat, lon] for lon, lat in route.geometry.coords]
                
                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=self._create_route_popup(route),
                    tooltip=f"{vehicle_id}: {route.get('num_stops', 0)} stops, {route.get('total_distance', 0):.0f}m"
                ).add_to(route_group)
                
                # Add start marker
                if route_coords:
                    folium.Marker(
                        location=route_coords[0],
                        popup=f"🏁 Start: {vehicle_id}",
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(route_group)
                    
                    # Add end marker
                    folium.Marker(
                        location=route_coords[-1],
                        popup=f"🏁 End: {vehicle_id}",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(route_group)
            
            # Add direction arrows if available
            if hasattr(route, 'directions') and route.directions:
                self._add_direction_arrows(route_group, route, color)
            
            route_group.add_to(map_obj)
    
    def _add_building_markers(self, map_obj: folium.Map, buildings_gdf: gpd.GeoDataFrame):
        """Add building markers grouped by cluster."""
        
        building_group = folium.FeatureGroup(name="🏠 Buildings")
        
        for cluster_id in buildings_gdf['cluster'].unique():
            cluster_buildings = buildings_gdf[buildings_gdf['cluster'] == cluster_id]
            color = self.color_palette[cluster_id % len(self.color_palette)]
            
            for idx, building in cluster_buildings.iterrows():
                if building.geometry is not None:
                    coords = [building.geometry.y, building.geometry.x]
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=5,
                        popup=f"Building {idx}<br>Cluster: {cluster_id}<br>Snap distance: {building.get('snap_distance', 0):.1f}m",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(building_group)
        
        building_group.add_to(map_obj)
    
    def _add_direction_arrows(self, feature_group: folium.FeatureGroup, route: pd.Series, color: str):
        """Add direction arrows along the route."""
        if not hasattr(route, 'directions') or not route.directions:
            return
        
        for step in route.directions:
            if step.get('geometry') and step['geometry'].get('coordinates'):
                coords = step['geometry']['coordinates']
                if len(coords) >= 2:
                    # Add arrow at midpoint of step
                    mid_idx = len(coords) // 2
                    mid_coord = [coords[mid_idx][1], coords[mid_idx][0]]  # lat, lon
                    
                    emoji = step.get('emoji', '🟢')
                    instruction = step.get('instruction', 'Continue')
                    
                    folium.Marker(
                        location=mid_coord,
                        popup=f"{emoji} {instruction}<br>Distance: {step.get('distance_m', 0):.0f}m",
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 12px;">{emoji}</div>',
                            icon_size=(20, 20),
                            icon_anchor=(10, 10)
                        )
                    ).add_to(feature_group)
    
    def _create_route_popup(self, route: pd.Series) -> str:
        """Create detailed popup for route."""
        popup_html = f"""
        <div style="width: 200px;">
            <h4>🚛 {route.get('vehicle_id', 'Unknown Vehicle')}</h4>
            <p><strong>Type:</strong> {route.get('vehicle_type', 'standard')}</p>
            <p><strong>Capacity:</strong> {route.get('capacity', 0)} kg</p>
            <p><strong>Stops:</strong> {route.get('num_stops', 0)}</p>
            <p><strong>Distance:</strong> {route.get('total_distance', 0):.0f} m</p>
            <p><strong>Duration:</strong> {route.get('total_duration', 0) / 60:.1f} min</p>
        """
        
        if hasattr(route, 'num_turns'):
            popup_html += f"""
            <hr>
            <p><strong>🟢 Straight:</strong> {route.get('num_straight', 0)}</p>
            <p><strong>🟡 Turns:</strong> {route.get('num_turns', 0)}</p>
            <p><strong>🔴 Complex:</strong> {route.get('num_complex_maneuvers', 0)}</p>
            """
        
        popup_html += "</div>"
        return popup_html
    
    def _add_legend(self, map_obj: folium.Map, routes_gdf: gpd.GeoDataFrame):
        """Add legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>🗺️ Route Legend</h4>
        <p><span style="color:green;">🟢</span> Straight / Continue</p>
        <p><span style="color:blue;">🔵</span> Slight Turn</p>
        <p><span style="color:orange;">🟡</span> Turn</p>
        <p><span style="color:red;">🔴</span> Sharp Turn / U-turn</p>
        <hr>
        <p><strong>Total Routes:</strong> {}</p>
        <p><strong>Total Distance:</strong> {:.1f} km</p>
        </div>
        '''.format(
            len(routes_gdf),
            routes_gdf['total_distance'].sum() / 1000 if 'total_distance' in routes_gdf.columns else 0
        )
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def save_map(self, map_obj: folium.Map, output_path: str = "route_map.html") -> str:
        """Save map to HTML file."""
        map_obj.save(output_path)
        logger.info(f"Saved interactive map to {output_path}")
        return output_path
    
    def create_cluster_analysis_map(self, buildings_gdf: gpd.GeoDataFrame, 
                                  center_coords: tuple = None) -> folium.Map:
        """Create map showing building clusters before routing."""
        
        if center_coords is None:
            bounds = buildings_gdf.total_bounds
            center_coords = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        m = folium.Map(location=center_coords, zoom_start=12)
        
        # Add clusters
        for cluster_id in buildings_gdf['cluster'].unique():
            cluster_buildings = buildings_gdf[buildings_gdf['cluster'] == cluster_id]
            color = self.color_palette[cluster_id % len(self.color_palette)]
            
            cluster_group = folium.FeatureGroup(name=f"Cluster {cluster_id} ({len(cluster_buildings)} buildings)")
            
            for idx, building in cluster_buildings.iterrows():
                if building.geometry is not None:
                    coords = [building.geometry.y, building.geometry.x]
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=6,
                        popup=f"Building {idx}<br>Cluster: {cluster_id}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.8
                    ).add_to(cluster_group)
            
            cluster_group.add_to(m)
        
        folium.LayerControl().add_to(m)
        logger.info(f"Created cluster analysis map with {len(buildings_gdf['cluster'].unique())} clusters")
        return m