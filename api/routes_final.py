from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import geopandas as gpd
import folium
from sklearn.cluster import KMeans, DBSCAN
import json
import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist
import math
import pandas as pd
from io import BytesIO, StringIO
import requests
import osmnx as ox
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Smart Waste Management System",
    description="Intelligent Garbage Collection Route Assignment System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store uploaded data
ward_gdf = None
roads_gdf = None
buildings_gdf = None

@app.post("/upload", summary="Upload GeoJSON and CSV files", description="Upload ward boundaries, roads, houses GeoJSON files and vehicles CSV file")
async def upload_files(
    ward_geojson: UploadFile = File(..., description="Ward boundaries GeoJSON file"),
    roads_geojson: UploadFile = File(..., description="Roads network GeoJSON file"),
    houses_geojson: UploadFile = File(..., description="Houses/buildings GeoJSON file"),
    vehicles_csv: UploadFile = File(..., description="Vehicles information CSV file")
):
    global ward_gdf, roads_gdf, buildings_gdf
    
    try:
        ward_content = await ward_geojson.read()
        roads_content = await roads_geojson.read()
        houses_content = await houses_geojson.read()
        
        ward_gdf = gpd.read_file(BytesIO(ward_content))
        roads_gdf = gpd.read_file(BytesIO(roads_content))
        buildings_gdf = gpd.read_file(BytesIO(houses_content))
        
        return {
            "status": "success", 
            "message": "Files uploaded successfully",
            "ward_features": len(ward_gdf),
            "roads_features": len(roads_gdf),
            "buildings_features": len(buildings_gdf)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")

@app.get("/clusters", summary="Get cluster coordinates", description="Get longitude and latitude for each cluster separately")
async def get_clusters():
    """Return cluster coordinates as separate response bodies."""
    if ward_gdf is None or buildings_gdf is None:
        raise HTTPException(status_code=400, detail="Please upload data files first using /upload endpoint")
    
    try:
        # Convert to WGS84 if needed
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf_wgs = buildings_gdf.to_crs('EPSG:4326')
        else:
            buildings_gdf_wgs = buildings_gdf
        
        # Cluster buildings using same logic as map generation
        buildings_utm = buildings_gdf_wgs.to_crs('EPSG:32644')
        building_centroids = np.array([(pt.x, pt.y) for pt in buildings_utm.geometry.centroid])
        
        # Use DBSCAN for initial clustering, then refine with K-means
        dbscan = DBSCAN(eps=200, min_samples=2)
        initial_clusters = dbscan.fit_predict(building_centroids)
        
        # If DBSCAN creates too many clusters, use K-means
        if len(set(initial_clusters)) > 5 or -1 in initial_clusters:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            building_clusters = kmeans.fit_predict(building_centroids)
        else:
            building_clusters = initial_clusters
        
        # Convert building centroids to WGS84
        building_centroids_wgs84 = [(pt.x, pt.y) for pt in buildings_utm.to_crs('EPSG:4326').geometry.centroid]
        
        # Get center coordinates for depot calculation
        bounds = buildings_gdf_wgs.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Define depot locations for each vehicle
        depot_locations = [
            (center_lon - 0.002, center_lat - 0.002),
            (center_lon + 0.002, center_lat - 0.002),
            (center_lon, center_lat),
            (center_lon - 0.002, center_lat + 0.002),
            (center_lon + 0.002, center_lat + 0.002)
        ]
        
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
        
        # Create cluster response with routes
        clusters = {}
        for cluster_id in range(5):
            cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
            
            if cluster_buildings:
                cluster_coords = [building_centroids_wgs84[i] for i in cluster_buildings]
                depot = depot_locations[cluster_id]
                
                # Create optimized route for cluster
                if len(cluster_coords) > 1:
                    optimized_route = [cluster_coords[0]]
                    remaining = cluster_coords[1:]
                    current = cluster_coords[0]
                    
                    while remaining:
                        distances = [((current[0]-pt[0])**2 + (current[1]-pt[1])**2)**0.5 for pt in remaining]
                        nearest_idx = np.argmin(distances)
                        next_pt = remaining.pop(nearest_idx)
                        optimized_route.append(next_pt)
                        current = next_pt
                else:
                    optimized_route = cluster_coords
                
                # Complete route: depot -> houses -> depot
                complete_route = [depot] + optimized_route + [depot]
                
                clusters[f"cluster_{cluster_id + 1}"] = {
                    "vehicle": vehicle_names[cluster_id],
                    "starting_point": {"longitude": depot[0], "latitude": depot[1]},
                    "ending_point": {"longitude": depot[0], "latitude": depot[1]},
                    "coordinates": [
                        {"longitude": coord[0], "latitude": coord[1]} 
                        for coord in cluster_coords
                    ],
                    "route": [
                        {"longitude": coord[0], "latitude": coord[1]} 
                        for coord in complete_route
                    ],
                    "count": len(cluster_coords)
                }
        
        return JSONResponse(clusters)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting clusters: {str(e)}")

@app.get("/generate-map/", response_class=HTMLResponse, summary="Generate optimized route map", description="Generate an interactive map with optimized garbage collection routes")
async def generate_map():
    # Check if data has been uploaded
    if ward_gdf is None or roads_gdf is None or buildings_gdf is None:
        raise HTTPException(status_code=400, detail="Please upload data files first using /upload endpoint")
    # Convert to WGS84 if needed
    if ward_gdf.crs != 'EPSG:4326':
        ward_gdf_wgs = ward_gdf.to_crs('EPSG:4326')
    else:
        ward_gdf_wgs = ward_gdf
        
    if roads_gdf.crs != 'EPSG:4326':
        roads_gdf_wgs = roads_gdf.to_crs('EPSG:4326')
    else:
        roads_gdf_wgs = roads_gdf
        
    if buildings_gdf.crs != 'EPSG:4326':
        buildings_gdf_wgs = buildings_gdf.to_crs('EPSG:4326')
    else:
        buildings_gdf_wgs = buildings_gdf
    
    # Get center coordinates from ward bounds
    bounds = ward_gdf_wgs.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Download OSM road network for the area
    try:
        G = ox.graph_from_bbox(bounds[3], bounds[1], bounds[2], bounds[0], network_type='drive')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except:
        G = None
    
    # Enhanced clustering using DBSCAN for better spatial distribution
    buildings_utm = buildings_gdf_wgs.to_crs('EPSG:32644')
    building_centroids = np.array([(pt.x, pt.y) for pt in buildings_utm.geometry.centroid])
    
    # Use DBSCAN for initial clustering, then refine with K-means
    dbscan = DBSCAN(eps=200, min_samples=2)
    initial_clusters = dbscan.fit_predict(building_centroids)
    
    # If DBSCAN creates too many clusters, use K-means
    if len(set(initial_clusters)) > 5 or -1 in initial_clusters:
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        building_clusters = kmeans.fit_predict(building_centroids)
    else:
        building_clusters = initial_clusters
    
    # OR-Tools VRP solver for route optimization
    def solve_vrp(locations, depot_index=0):
        """Solve Vehicle Routing Problem using OR-Tools"""
        if len(locations) <= 1:
            return [0]
            
        # Create distance matrix
        def compute_distance_matrix(locations):
            distances = []
            for from_counter, from_node in enumerate(locations):
                distances.append([])
                for to_counter, to_node in enumerate(locations):
                    if from_counter == to_counter:
                        distances[from_counter].append(0)
                    else:
                        # Euclidean distance in meters
                        dist = int(np.sqrt((from_node[0] - to_node[0])**2 + (from_node[1] - to_node[1])**2))
                        distances[from_counter].append(dist)
            return distances
        
        distance_matrix = compute_distance_matrix(locations)
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, depot_index)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
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
        return list(range(len(locations)))
    
    # Enhanced OSRM routing with turn-by-turn directions
    def get_enhanced_osrm_route(start_coord, end_coord):
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{start_coord[0]},{start_coord[1]};{end_coord[0]},{end_coord[1]}?overview=full&geometries=geojson&steps=true&annotations=true"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['routes']:
                    route = data['routes'][0]
                    coords = [[coord[1], coord[0]] for coord in route['geometry']['coordinates']]
                    
                    # Extract turn-by-turn directions
                    directions = []
                    if 'legs' in route:
                        for leg in route['legs']:
                            if 'steps' in leg:
                                for step in leg['steps']:
                                    if 'maneuver' in step:
                                        instruction = step['maneuver'].get('instruction', 'Continue')
                                        distance = step.get('distance', 0)
                                        duration = step.get('duration', 0)
                                        directions.append({
                                            'instruction': instruction,
                                            'distance': f"{distance:.0f}m",
                                            'duration': f"{duration:.0f}s"
                                        })
                    
                    return coords, directions, route.get('distance', 0), route.get('duration', 0)
        except:
            pass
        return [[start_coord[1], start_coord[0]], [end_coord[1], end_coord[0]]], [], 0, 0
    
    # Create map with custom tiles
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add custom map layers
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add ward boundaries
    folium.GeoJson(
        json.loads(ward_gdf_wgs[['geometry']].to_json()),
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'darkblue',
            'weight': 3,
            'fillOpacity': 0
        }
    ).add_to(m)
    
    # Colors and vehicle names
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
    
    # Convert building centroids to WGS84
    building_centroids_wgs84 = [(pt.x, pt.y) for pt in buildings_utm.to_crs('EPSG:4326').geometry.centroid]
    
    # Process each cluster with enhanced routing
    route_info = {}
    
    for cluster_id in range(5):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        
        if not cluster_buildings:
            continue
            
        # Get house locations for this cluster
        house_locations_wgs84 = [building_centroids_wgs84[i] for i in cluster_buildings]
        house_locations_utm = [building_centroids[i] for i in cluster_buildings]
        
        # Optimize route using OR-Tools VRP
        optimized_route = solve_vrp(house_locations_utm)
        optimized_locations = [house_locations_wgs84[i] for i in optimized_route]
        
        # Create enhanced route with OSRM
        all_route_coords = []
        total_distance = 0
        total_duration = 0
        all_directions = []
        
        for i in range(len(optimized_locations) - 1):
            start_pt = optimized_locations[i]
            end_pt = optimized_locations[i + 1]
            
            coords, directions, distance, duration = get_enhanced_osrm_route(start_pt, end_pt)
            all_route_coords.extend(coords)
            all_directions.extend(directions)
            total_distance += distance
            total_duration += duration
        
        # Store route information
        route_info[cluster_id] = {
            'distance': f"{total_distance/1000:.2f} km",
            'duration': f"{total_duration/60:.0f} min",
            'houses': len(cluster_buildings),
            'directions': all_directions[:10]  # Limit to first 10 directions
        }
        
        # Add optimized route to map
        if all_route_coords:
            folium.PolyLine(
                all_route_coords,
                color=colors[cluster_id],
                weight=4,
                opacity=0.8,
                popup=f"{vehicle_names[cluster_id]}<br>Distance: {route_info[cluster_id]['distance']}<br>Duration: {route_info[cluster_id]['duration']}<br>Houses: {route_info[cluster_id]['houses']}"
            ).add_to(m)
        
        # Add start marker
        if optimized_locations:
            start_pt = optimized_locations[0]
            folium.Marker(
                [start_pt[1], start_pt[0]],
                popup=f"{vehicle_names[cluster_id]} Start<br>{route_info[cluster_id]['houses']} houses<br>{route_info[cluster_id]['distance']}, {route_info[cluster_id]['duration']}",
                icon=folium.Icon(color='darkgreen', icon='play')
            ).add_to(m)
            
            # Add end marker if different from start
            if len(optimized_locations) > 1:
                end_pt = optimized_locations[-1]
                if abs(end_pt[0] - start_pt[0]) > 0.0001 or abs(end_pt[1] - start_pt[1]) > 0.0001:
                    folium.Marker(
                        [end_pt[1], end_pt[0]],
                        popup=f"{vehicle_names[cluster_id]} End",
                        icon=folium.Icon(color='darkred', icon='stop')
                    ).add_to(m)
        
        # Add house markers with sequence numbers
        for i, house_idx in enumerate(optimized_route[:-1], 1):
            if house_idx < len(house_locations_wgs84):
                house_pt = house_locations_wgs84[house_idx]
                folium.CircleMarker(
                    [house_pt[1], house_pt[0]],
                    radius=6,
                    popup=f"{vehicle_names[cluster_id]} - Stop {i}",
                    color='white',
                    fillColor=colors[cluster_id],
                    fillOpacity=0.9,
                    weight=2
                ).add_to(m)
                
                # Add sequence number
                folium.Marker(
                    [house_pt[1], house_pt[0]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 10px; color: white; font-weight: bold; text-align: center; line-height: 15px;">{i}</div>',
                        icon_size=(15, 15),
                        icon_anchor=(7, 7)
                    )
                ).add_to(m)
        
        # Add building polygons
        for house_number, building_idx in enumerate(cluster_buildings, 1):
            building = buildings_gdf_wgs.iloc[building_idx]
            folium.GeoJson(
                json.loads(gpd.GeoSeries([building.geometry]).to_json()),
                style_function=lambda x, c=colors[cluster_id]: {
                    'fillColor': c,
                    'color': c,
                    'weight': 1,
                    'fillOpacity': 0.6
                },
                popup=f"{vehicle_names[cluster_id]} - House {house_number}",
                tooltip=f"C{cluster_id + 1}-H{house_number}"
            ).add_to(m)
    
    # Enhanced legend with route information
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: auto; 
                background-color: rgba(255,255,255,0.95); border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px;">🚗 Vehicle Routes</p>
    '''
    
    for i, (color, vehicle) in enumerate(zip(colors, vehicle_names)):
        if i in route_info:
            info = route_info[i]
            legend_html += f'''
            <div style="margin: 5px 0; padding: 3px; border-left: 4px solid {color};">
                <div style="font-weight: bold;">{vehicle}</div>
                <div>📏 {info['distance']} | ⏱️ {info['duration']}</div>
                <div>🏠 {info['houses']} houses</div>
            </div>
            '''
        else:
            legend_html += f'<p style="margin: 3px 0; color: #888;"><i style="background:{color}; width:15px; height:10px; display:inline-block; margin-right: 5px; border-radius: 2px;"></i> {vehicle} - No route</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return HTMLResponse(content=m._repr_html_())

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>🚛 Smart Waste Management</title>
        </head>
        <body>
            <h2>🚛 Smart Waste Management System</h2>
            <div style="margin: 20px 0;">
                <h3>Steps:</h3>
                <ol>
                    <li>First upload your data files using <a href="/docs#/default/upload_files_upload_post">/upload</a></li>
                    <li>Then <a href="/generate-map/">🗺️ Generate Optimized Routes</a></li>
                </ol>
            </div>
            <p><a href="/docs">📚 API Documentation</a></p>
        </body>
    </html>
    """)