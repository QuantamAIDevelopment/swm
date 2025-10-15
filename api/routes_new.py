"""FastAPI routes for the garbage collection system."""
import logging
import json
from datetime import datetime
from typing import Dict, Any
from io import StringIO, BytesIO
import geopandas as gpd
import pandas as pd
import folium
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Garbage Collection Route Assignment System")

# Global storage for uploaded data
current_data = None

@app.post("/upload")
async def upload_files(
    ward_geojson: UploadFile = File(...),
    roads_geojson: UploadFile = File(...),
    houses_geojson: UploadFile = File(...),
    vehicles_csv: UploadFile = File(...)
):
    """Upload input files for route computation."""
    global current_data
    try:
        # Read uploaded files
        ward_content = await ward_geojson.read()
        roads_content = await roads_geojson.read()
        houses_content = await houses_geojson.read()
        vehicles_content = await vehicles_csv.read()
        
        # Parse GeoJSON files
        ward_boundaries = gpd.read_file(BytesIO(ward_content))
        road_network = gpd.read_file(BytesIO(roads_content))
        houses = gpd.read_file(BytesIO(houses_content))
        
        # Parse CSV file
        vehicles_str = vehicles_content.decode('utf-8')
        vehicles = pd.read_csv(StringIO(vehicles_str))
        
        # Store data globally
        current_data = {
            "ward_boundaries": ward_boundaries,
            "road_network": road_network,
            "houses": houses,
            "vehicles": vehicles,
            "timestamp": datetime.now()
        }
        
        logger.info("Files uploaded successfully")
        return {"status": "success", "message": "Files uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/map")
async def get_map():
    """Generate K-means clustered map with vehicle routes."""
    global current_data
    try:
        if current_data is None:
            m = folium.Map(location=[17.385, 78.486], zoom_start=12)
            folium.Marker(
                [17.385, 78.486],
                popup="Please upload files first",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            return HTMLResponse(content=m._repr_html_())
        
        # Load and convert data
        ward_gdf = current_data["ward_boundaries"].to_crs('EPSG:4326')
        roads_gdf = current_data["road_network"].to_crs('EPSG:4326')
        buildings_gdf = current_data["houses"].to_crs('EPSG:4326')
        
        # Get center coordinates
        bounds = ward_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        
        # Add ward boundaries (dark blue)
        for idx, ward in ward_gdf.iterrows():
            folium.GeoJson(
                ward.geometry,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'darkblue',
                    'weight': 3,
                    'fillOpacity': 0
                }
            ).add_to(m)
        
        # Add road networks (black)
        for idx, road in roads_gdf.iterrows():
            folium.GeoJson(
                road.geometry,
                style_function=lambda x: {
                    'color': 'black',
                    'weight': 2
                }
            ).add_to(m)
        
        # K-means clustering on building centroids
        from sklearn.cluster import KMeans
        import networkx as nx
        import numpy as np
        import math
        from tools.directions_generator import DirectionsGenerator
        
        # Initialize directions generator
        directions_gen = DirectionsGenerator()
        
        building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        building_clusters = kmeans.fit_predict(building_centroids)
        
        # Create road network graph
        G = nx.Graph()
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
            else:
                coords = list(geom.coords)
                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                    G.add_edge(p1, p2, weight=dist)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
        
        # Process each cluster
        for cluster_id in range(5):
            cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
            
            if not cluster_buildings:
                continue
            
            # Get all road points
            cluster_road_points = []
            for idx, road in roads_gdf.iterrows():
                geom = road.geometry
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        cluster_road_points.extend(list(line.coords))
                else:
                    cluster_road_points.extend(list(geom.coords))
            
            if cluster_buildings:
                cluster_road_points = list(set(cluster_road_points))
                house_locations_wgs84 = []
                for i in cluster_buildings:
                    pt = buildings_gdf.iloc[i].geometry.centroid
                    house_locations_wgs84.append((pt.x, pt.y))
                
                # Find nearest road points to houses
                house_road_points = []
                for house_pt in house_locations_wgs84:
                    distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                    nearest_idx = np.argmin(distances)
                    house_road_points.append(cluster_road_points[nearest_idx])
                
                # Create collection route
                if house_road_points and len(house_road_points) > 0:
                    unique_house_points = []
                    for pt in house_road_points:
                        if pt not in unique_house_points:
                            unique_house_points.append(pt)
                    
                    if len(unique_house_points) >= 1:
                        start_point = unique_house_points[0]
                        route_points = [start_point]
                        current_point = start_point
                        remaining_points = unique_house_points[1:].copy()
                        
                        while remaining_points:
                            distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                            nearest_idx = np.argmin(distances)
                            next_point = remaining_points.pop(nearest_idx)
                            
                            try:
                                path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                                route_points.extend(path_segment[1:])
                            except:
                                route_points.append(next_point)
                            
                            current_point = next_point
                        
                        # Convert to lat/lon for folium
                        route_coords = [[pt[1], pt[0]] for pt in route_points]
                        
                        # Generate flow-based directions using route geometry
                        if hasattr(route, 'geometry') and route.geometry:
                            # Use actual route geometry coordinates
                            route_flow_coords = [(coord[1], coord[0]) for coord in route.geometry.coords]
                        else:
                            # Use constructed route coordinates
                            route_flow_coords = [(coord[0], coord[1]) for coord in route_coords]
                        
                        directions = directions_gen.generate_route_flow_directions(
                            route_flow_coords,
                            start_point=(start_point[1], start_point[0]),
                            end_point=(current_point[1], current_point[0]),
                            vehicle_names[cluster_id]
                        )
                        
                        # Create directions text for popup
                        directions_text = directions_gen.format_directions_text(directions)
                        route_summary = directions_gen.generate_route_summary(directions, vehicle_names[cluster_id])
                        
                        # Add collection route with directions popup
                        folium.PolyLine(
                            route_coords,
                            color=colors[cluster_id],
                            weight=4,
                            opacity=0.8,
                            popup=folium.Popup(
                                f"<b>{vehicle_names[cluster_id]}</b><br>"
                                f"Houses: {len(cluster_buildings)}<br>"
                                f"Distance: {route_summary.get('total_distance_km', 0)}km<br>"
                                f"Steps: {route_summary.get('total_steps', 0)}<br><br>"
                                f"<b>Directions:</b><br>"
                                f"<div style='max-height:200px;overflow-y:auto;font-size:11px;'>"
                                f"{directions_text.replace(chr(10), '<br>')}</div>",
                                max_width=400
                            )
                        ).add_to(m)
                        
                        # Add directional arrows with turn indicators
                        for i, direction in enumerate(directions[1:-1], 1):  # Skip start and end
                            if i < len(route_coords):
                                lat, lon = route_coords[i]
                                dir_type = direction['direction']
                                
                                # Choose arrow symbol based on direction
                                if 'left' in dir_type:
                                    arrow = '↰' if 'slight' in dir_type else '←'
                                elif 'right' in dir_type:
                                    arrow = '↱' if 'slight' in dir_type else '→'
                                elif 'u-turn' in dir_type.lower():
                                    arrow = '↶'
                                else:
                                    arrow = '↑'
                                
                                folium.Marker(
                                    [lat, lon],
                                    icon=folium.DivIcon(
                                        html=f'<div style="color: {colors[cluster_id]}; font-size: 18px; font-weight: bold;">{arrow}</div>',
                                        icon_size=(20, 20),
                                        icon_anchor=(10, 10)
                                    ),
                                    popup=f"Step {direction['step']}: {direction['instruction']}"
                                ).add_to(m)
                        
                        # Add start/end markers with summary
                        folium.Marker(
                            [start_point[1], start_point[0]],
                            popup=folium.Popup(
                                f"<b>{vehicle_names[cluster_id]} START</b><br>"
                                f"Total Distance: {route_summary.get('total_distance_km', 0)}km<br>"
                                f"Houses to collect: {len(cluster_buildings)}<br>"
                                f"Total steps: {route_summary.get('total_steps', 0)}",
                                max_width=250
                            ),
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(m)
                        
                        folium.Marker(
                            [current_point[1], current_point[0]],
                            popup=folium.Popup(
                                f"<b>{vehicle_names[cluster_id]} END</b><br>"
                                f"Route completed<br>"
                                f"Total distance: {route_summary.get('total_distance_km', 0)}km",
                                max_width=250
                            ),
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(m)
                        
                        # Add house markers
                        for i, house_pt in enumerate(house_locations_wgs84, 1):
                            folium.CircleMarker(
                                [house_pt[1], house_pt[0]],
                                radius=3,
                                popup=f"{vehicle_names[cluster_id]} - House {i}",
                                color=colors[cluster_id],
                                fillColor=colors[cluster_id],
                                fillOpacity=0.8
                            ).add_to(m)
            
            # Add clustered buildings as polygons
            for house_number, building_idx in enumerate(cluster_buildings, 1):
                building = buildings_gdf.iloc[building_idx]
                folium.GeoJson(
                    building.geometry,
                    style_function=lambda x, c=colors[cluster_id]: {
                        'fillColor': c,
                        'color': c,
                        'weight': 1,
                        'fillOpacity': 0.6
                    },
                    popup=f"{vehicle_names[cluster_id]} - House {house_number}",
                    tooltip=f"C{cluster_id + 1}-H{house_number}"
                ).add_to(m)
        
        return HTMLResponse(content=m._repr_html_())
        
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        m = folium.Map(location=[17.385, 78.486], zoom_start=12)
        folium.Marker(
            [17.385, 78.486],
            popup=f"Error: {str(e)}",
            icon=folium.Icon(color='red', icon='exclamation-sign')
        ).add_to(m)
        return HTMLResponse(content=m._repr_html_())

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return HTMLResponse(content="""
    <html>
        <body>
            <h2>Ward Map Generator</h2>
            <p><a href="/map">Generate Map</a></p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)