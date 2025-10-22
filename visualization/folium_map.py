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
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF',
            '#FF0080', '#00FFFF', '#FFFF00', '#FF4000', '#4000FF',
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
        
        # Add comprehensive cluster panel with route information
        self._add_cluster_panel(m, buildings_gdf, routes_gdf)
        
        # Add enhanced layer control on the right side
        layer_control = folium.LayerControl(
            position='topright',
            collapsed=False,
            autoZIndex=True
        )
        layer_control.add_to(m)
        
        # Add custom CSS for better layer control styling
        self._add_layer_control_styling(m)
        
        logger.info(f"Created Folium map with {len(routes_gdf)} routes")
        return m
    
    def _add_route_layers(self, map_obj: folium.Map, routes_gdf: gpd.GeoDataFrame):
        """Add route layers to map with separate toggleable layers per cluster."""
        
        for idx, route in routes_gdf.iterrows():
            cluster_id = int(route.get('cluster_id', idx))
            color = self.color_palette[cluster_id % len(self.color_palette)]
            vehicle_id = route.get('vehicle_id', f'Vehicle {cluster_id}')
            num_stops = route.get('num_stops', 0)
            distance = route.get('total_distance', 0) / 1000  # Convert to km
            
            # Create separate layer for each cluster
            route_group = folium.FeatureGroup(
                name=f"🚛 Cluster {cluster_id} ({num_stops} stops, {distance:.1f}km)",
                show=True  # All layers visible by default
            )
            
            if route.geometry is not None and hasattr(route.geometry, 'coords'):
                coords_list = list(route.geometry.coords)
                if len(coords_list) >= 2:
                    route_coords = [[lat, lon] for lon, lat in coords_list]
                    
                    # Add route line
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=5,
                        opacity=0.8,
                        popup=self._create_route_popup(route),
                        tooltip=f"Cluster {cluster_id}: {num_stops} stops, {distance:.1f}km"
                    ).add_to(route_group)
                    
                    # Add start marker
                    folium.Marker(
                        location=route_coords[0],
                        popup=f"🏁 Start - Cluster {cluster_id}<br>{vehicle_id}",
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(route_group)
                    
                    # Add end marker
                    folium.Marker(
                        location=route_coords[-1],
                        popup=f"🏁 End - Cluster {cluster_id}<br>{vehicle_id}",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(route_group)
            
            route_group.add_to(map_obj)
    
    def _add_building_markers(self, map_obj: folium.Map, buildings_gdf: gpd.GeoDataFrame):
        """Add building markers with separate layers per cluster."""
        
        for cluster_id in sorted(buildings_gdf['cluster'].unique()):
            cluster_buildings = buildings_gdf[buildings_gdf['cluster'] == cluster_id]
            color = self.color_palette[cluster_id % len(self.color_palette)]
            
            # Create separate building layer for each cluster
            building_group = folium.FeatureGroup(
                name=f"🏠 Buildings - Cluster {cluster_id} ({len(cluster_buildings)})",
                show=False  # Buildings hidden by default to reduce clutter
            )
            
            for idx, building in cluster_buildings.iterrows():
                if building.geometry is not None:
                    coords = [building.geometry.y, building.geometry.x]
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=4,
                        popup=f"Building {idx}<br>Cluster: {cluster_id}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(building_group)
            
            building_group.add_to(map_obj)
    
    def _create_route_popup(self, route: pd.Series) -> str:
        """Create detailed popup for route."""
        popup_html = f"""
        <div style="width: 200px;">
            <h4>🚛 {route.get('vehicle_id', 'Unknown Vehicle')}</h4>
            <p><strong>Cluster:</strong> {route.get('cluster_id', 'N/A')}</p>
            <p><strong>Type:</strong> {route.get('vehicle_type', 'standard')}</p>
            <p><strong>Capacity:</strong> {route.get('capacity', 0)} kg</p>
            <p><strong>Stops:</strong> {route.get('num_stops', 0)}</p>
            <p><strong>Distance:</strong> {route.get('total_distance', 0):.0f} m</p>
            <p><strong>Duration:</strong> {route.get('total_duration', 0) / 60:.1f} min</p>
        </div>
        """
        return popup_html
    
    def _add_legend(self, map_obj: folium.Map, routes_gdf: gpd.GeoDataFrame):
        """Add legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>🗺️ Route Legend</h4>
        <p><span style="color:green;">🟢</span> Cluster Start</p>
        <p><span style="color:red;">🔴</span> Cluster End</p>
        <hr>
        <p><strong>Total Routes:</strong> {}</p>
        <p><strong>Total Distance:</strong> {:.1f} km</p>
        </div>
        '''.format(
            len(routes_gdf),
            routes_gdf['total_distance'].sum() / 1000 if 'total_distance' in routes_gdf.columns else 0
        )
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_cluster_panel(self, map_obj: folium.Map, buildings_gdf: gpd.GeoDataFrame, routes_gdf: gpd.GeoDataFrame = None):
        """Add cluster information panel on the right side."""
        cluster_counts = buildings_gdf['cluster'].value_counts().sort_index()
        
        cluster_items = []
        for cluster_id in sorted(cluster_counts.index):
            color = self.color_palette[cluster_id % len(self.color_palette)]
            count = cluster_counts[cluster_id]
            
            route_info = ""
            if routes_gdf is not None:
                cluster_route = routes_gdf[routes_gdf['cluster_id'] == cluster_id]
                if not cluster_route.empty:
                    route = cluster_route.iloc[0]
                    distance = route.get('total_distance', 0) / 1000
                    duration = route.get('total_duration', 0) / 60
                    vehicle = route.get('vehicle_id', f'V{cluster_id}')
                    route_info = f"<br><small>{vehicle} • {distance:.1f}km • {duration:.0f}min</small>"
            
            cluster_items.append(f'<div style="margin:5px 0;padding:8px;border:1px solid #ddd;border-radius:4px;"><span style="color:{color};font-size:14px;">●</span> <strong>Cluster {cluster_id}</strong><br>{count} buildings{route_info}</div>')
        
        panel_html = f'''
        <div style="position:fixed;top:10px;right:10px;width:250px;max-height:70vh;background:white;border:2px solid #333;z-index:9999;font-size:12px;border-radius:5px;box-shadow:0 2px 10px rgba(0,0,0,0.3);">
            <div style="background:#333;color:white;padding:8px;border-radius:3px 3px 0 0;">
                <strong>📊 Cluster Dashboard</strong>
                <div style="font-size:10px;margin-top:3px;">{len(cluster_counts)} clusters • {len(buildings_gdf)} buildings</div>
            </div>
            <div style="padding:8px;max-height:50vh;overflow-y:auto;">
                {''.join(cluster_items)}
            </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(panel_html))
    
    def _add_layer_control_styling(self, map_obj: folium.Map):
        """Add custom CSS styling."""
        style_html = '<style>.leaflet-control-layers{background:rgba(255,255,255,0.95)!important;border-radius:5px!important;}</style>'
        map_obj.get_root().html.add_child(folium.Element(style_html))
    
    def save_map(self, map_obj: folium.Map, output_path: str = "route_map.html") -> str:
        """Save map to HTML file."""
        map_obj.save(output_path)
        logger.info(f"Saved interactive map to {output_path}")
        return output_path
    
    def create_cluster_analysis_map(self, buildings_gdf: gpd.GeoDataFrame, 
                                  center_coords: tuple = None) -> folium.Map:
        """Create map showing building clusters with individual toggleable layers."""
        
        if center_coords is None:
            bounds = buildings_gdf.total_bounds
            center_coords = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        m = folium.Map(location=center_coords, zoom_start=12, tiles='OpenStreetMap')
        
        # Add tile layer options
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        for cluster_id in sorted(buildings_gdf['cluster'].unique()):
            cluster_buildings = buildings_gdf[buildings_gdf['cluster'] == cluster_id]
            color = self.color_palette[cluster_id % len(self.color_palette)]
            
            # Create toggleable layer for each cluster
            cluster_group = folium.FeatureGroup(
                name=f"📍 Cluster {cluster_id} ({len(cluster_buildings)} buildings)",
                show=True
            )
            
            for idx, building in cluster_buildings.iterrows():
                if building.geometry is not None:
                    coords = [building.geometry.y, building.geometry.x]
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=6,
                        popup=f"Building {idx}<br>Cluster: {cluster_id}<br>Lat: {coords[0]:.6f}<br>Lon: {coords[1]:.6f}",
                        tooltip=f"Cluster {cluster_id} - Building {idx}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(cluster_group)
            
            cluster_group.add_to(m)
        
        # Add enhanced layer control on the right side
        layer_control = folium.LayerControl(
            position='topright', 
            collapsed=False,
            autoZIndex=True
        )
        layer_control.add_to(m)
        
        # Add custom CSS for better layer control styling
        self._add_layer_control_styling(m)
        
        # Add comprehensive cluster panel
        self._add_cluster_panel(m, buildings_gdf)
        
        logger.info(f"Created cluster analysis map with {len(buildings_gdf['cluster'].unique())} toggleable cluster layers")
        return m