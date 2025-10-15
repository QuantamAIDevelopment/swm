"""FastAPI integration for geospatial route optimization."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import tempfile
import os
import shutil
import geopandas as gpd
import folium
from sklearn.cluster import KMeans
import json
import networkx as nx
import numpy as np
from shapely.geometry import Point
import math

app = FastAPI(
    title="Geospatial AI Route Optimizer",
    description="Dynamic garbage collection route optimization using road network data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/optimize-routes")
async def optimize_routes(
    roads_file: UploadFile = File(..., description="Roads GeoJSON file"),
    buildings_file: UploadFile = File(..., description="Buildings GeoJSON file"), 
    vehicles_file: UploadFile = File(..., description="Vehicles CSV file"),
    ward_geojson: UploadFile = File(..., description="Ward boundary GeoJSON file")
):
    """Upload files and run complete route optimization pipeline."""
    
    # Validate file types
    if not roads_file.filename or not roads_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Roads file must be GeoJSON")
    if not buildings_file.filename or not buildings_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Buildings file must be GeoJSON")
    if not vehicles_file.filename or not vehicles_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Vehicles file must be CSV")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded files
            roads_path = os.path.join(temp_dir, "roads.geojson")
            buildings_path = os.path.join(temp_dir, "buildings.geojson")
            vehicles_path = os.path.join(temp_dir, "vehicles.csv")
            ward_path = os.path.join(temp_dir, "ward.geojson")
            
            with open(roads_path, "wb") as f:
                shutil.copyfileobj(roads_file.file, f)
            with open(buildings_path, "wb") as f:
                shutil.copyfileobj(buildings_file.file, f)
            with open(vehicles_path, "wb") as f:
                shutil.copyfileobj(vehicles_file.file, f)
            with open(ward_path, "wb") as f:
                shutil.copyfileobj(ward_geojson.file, f)
            
            # Generate map using uploaded files
            map_html = generate_map_from_files(ward_path, roads_path, buildings_path)
            
            # Save map and data to output directory
            os.makedirs("output", exist_ok=True)
            with open("output/route_map.html", "w", encoding="utf-8") as f:
                f.write(map_html)
            
            # Save data files for cluster endpoint
            shutil.copy(ward_path, "output/ward.geojson")
            shutil.copy(buildings_path, "output/buildings.geojson")
            shutil.copy(roads_path, "output/roads.geojson")
            shutil.copy(vehicles_path, "output/vehicles.csv")
            
            return JSONResponse({
                "status": "success",
                "message": "Route optimization completed successfully",
                "maps": {
                    "route_map": "/generate-map/route_map"
                }
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/clusters", summary="Get cluster outlines with route coordinates", description="Get cluster boundaries and route coordinates for each vehicle cluster")
async def get_clusters():
    """Return cluster outlines with complete route coordinates using same logic as map generation."""
    try:
        # Check if files exist
        ward_path = "output/ward.geojson"
        buildings_path = "output/buildings.geojson"
        vehicles_path = "output/vehicles.csv"
        
        if not os.path.exists(ward_path) or not os.path.exists(buildings_path):
            raise HTTPException(status_code=404, detail="Data not found. Please upload files first using /optimize-routes")
        
        # Load vehicles data if available
        import pandas as pd
        vehicles_df = None
        active_vehicles = []
        if os.path.exists(vehicles_path):
            vehicles_df = pd.read_csv(vehicles_path)
            # Filter only active vehicles
            if 'status' in vehicles_df.columns:
                active_vehicles = vehicles_df[vehicles_df['status'].str.lower() == 'active'].reset_index(drop=True)
            else:
                active_vehicles = vehicles_df  # If no status column, assume all are active
        
        # Load GeoJSON data
        ward_gdf = gpd.read_file(ward_path)
        buildings_gdf = gpd.read_file(buildings_path)
        
        # Try to load roads data
        roads_gdf = None
        roads_path = "output/roads.geojson"
        if not os.path.exists(roads_path):
            for alt_path in ["roads.geojson", "output/road.geojson", "road.geojson"]:
                if os.path.exists(alt_path):
                    roads_path = alt_path
                    break
        
        if os.path.exists(roads_path):
            roads_gdf = gpd.read_file(roads_path)
        
        # Convert to WGS84 if needed
        if ward_gdf.crs != 'EPSG:4326':
            ward_gdf = ward_gdf.to_crs('EPSG:4326')
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        if roads_gdf is not None and roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        
        # Get center coordinates from ward bounds
        bounds = ward_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Use K-means clustering on building centroids (same as map generation)
        building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        
        # Determine number of clusters from active vehicles or default to 5
        n_clusters = len(active_vehicles) if len(active_vehicles) > 0 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        building_clusters = kmeans.fit_predict(building_centroids)
        
        # Create road network graph (same as map generation)
        G = nx.Graph()
        if roads_gdf is not None:
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
        
        # Define depot locations for each vehicle (same as map generation)
        depot_locations = []
        for i in range(n_clusters):
            if i == 0:
                depot_locations.append((center_lon - 0.002, center_lat - 0.002))
            elif i == 1:
                depot_locations.append((center_lon + 0.002, center_lat - 0.002))
            elif i == 2:
                depot_locations.append((center_lon, center_lat))
            elif i == 3:
                depot_locations.append((center_lon - 0.002, center_lat + 0.002))
            elif i == 4:
                depot_locations.append((center_lon + 0.002, center_lat + 0.002))
            else:
                # For additional vehicles, distribute around center
                angle = (i - 5) * (360 / max(1, n_clusters - 5))
                offset = 0.003
                depot_locations.append((
                    center_lon + offset * math.cos(math.radians(angle)),
                    center_lat + offset * math.sin(math.radians(angle))
                ))
        
        # Get all road points for routing
        cluster_road_points = []
        if roads_gdf is not None:
            for idx, road in roads_gdf.iterrows():
                geom = road.geometry
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        cluster_road_points.extend(list(line.coords))
                else:
                    cluster_road_points.extend(list(geom.coords))
        cluster_road_points = list(set(cluster_road_points))
        
        # Create cluster response with same routing logic as map generation
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
            
            # Always create cluster even if no buildings (for active vehicles)
            if True:  # Changed from 'if cluster_buildings:' to always create clusters
                # Get vehicle info from active vehicles CSV or use defaults
                if len(active_vehicles) > 0 and cluster_id < len(active_vehicles):
                    vehicle_row = active_vehicles.iloc[cluster_id]
                    vehicle_number = vehicle_row.get('vehicle_number', cluster_id + 1)
                    vehicle_id = vehicle_row.get('vehicle_id', f"VH_{cluster_id + 1:03d}")
                else:
                    vehicle_number = cluster_id + 1
                    vehicle_id = f"VH_{cluster_id + 1:03d}"
                
                depot_point = depot_locations[cluster_id]
                
                # Get house locations for this cluster (same as map generation)
                house_locations_wgs84 = []
                if cluster_buildings:  # Only if there are buildings in this cluster
                    for i in cluster_buildings:
                        pt = buildings_gdf.iloc[i].geometry.centroid
                        house_locations_wgs84.append((pt.x, pt.y))
                
                # Create cluster outline (convex hull)
                from shapely.geometry import MultiPoint
                if house_locations_wgs84:
                    cluster_points = MultiPoint(house_locations_wgs84 + [depot_point])
                    cluster_outline = list(cluster_points.convex_hull.exterior.coords)
                else:
                    # If no buildings, create small outline around depot
                    offset = 0.001
                    cluster_outline = [
                        (depot_point[0] - offset, depot_point[1] - offset),
                        (depot_point[0] + offset, depot_point[1] - offset),
                        (depot_point[0] + offset, depot_point[1] + offset),
                        (depot_point[0] - offset, depot_point[1] + offset),
                        (depot_point[0] - offset, depot_point[1] - offset)
                    ]
                
                # Generate route using same logic as map generation
                route_coordinates = []
                
                if cluster_road_points and house_locations_wgs84 and len(house_locations_wgs84) > 0:
                    # Find nearest road points to houses (same as map)
                    house_road_points = []
                    for house_pt in house_locations_wgs84:
                        distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                        nearest_idx = np.argmin(distances)
                        house_road_points.append(cluster_road_points[nearest_idx])
                    
                    # Remove duplicates while preserving order
                    unique_house_points = []
                    for pt in house_road_points:
                        if pt not in unique_house_points:
                            unique_house_points.append(pt)
                    
                    if len(unique_house_points) >= 1:
                        # Find nearest road point to depot
                        depot_distances = [((depot_point[0]-rp[0])**2 + (depot_point[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                        depot_road_point = cluster_road_points[np.argmin(depot_distances)]
                        
                        # Route: Depot -> Houses -> Depot (same as map)
                        route_points = [depot_road_point]
                        start_point = unique_house_points[0]
                        
                        # Go to first house
                        try:
                            path_to_first = nx.shortest_path(G, depot_road_point, start_point, weight='weight')
                            route_points.extend(path_to_first[1:])
                        except:
                            route_points.append(start_point)
                        
                        # Visit all other houses
                        current_point = start_point
                        remaining_points = unique_house_points[1:].copy()
                        
                        while remaining_points:
                            # Find nearest unvisited house
                            distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                            nearest_idx = np.argmin(distances)
                            next_point = remaining_points.pop(nearest_idx)
                            
                            # Find shortest path on roads
                            try:
                                path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                                route_points.extend(path_segment[1:])
                            except:
                                route_points.append(next_point)
                            
                            current_point = next_point
                        
                        # Return to depot
                        try:
                            path_to_depot = nx.shortest_path(G, current_point, depot_road_point, weight='weight')
                            route_points.extend(path_to_depot[1:])
                        except:
                            route_points.append(depot_road_point)
                        
                        # Convert to lat/lon coordinates
                        route_coordinates = [[pt[1], pt[0]] for pt in route_points]
                
                if not route_coordinates:
                    # Fallback: create route around depot or cluster outline
                    if house_locations_wgs84:
                        route_coordinates = [[coord[1], coord[0]] for coord in cluster_outline]
                    else:
                        # Just depot point if no buildings
                        route_coordinates = [[depot_point[1], depot_point[0]]]
                
                # Create road segments from route coordinates
                road_segments = []
                if len(route_coordinates) > 1:
                    for i in range(len(route_coordinates) - 1):
                        segment = {
                            "start": {
                                "longitude": route_coordinates[i][1],
                                "latitude": route_coordinates[i][0]
                            },
                            "end": {
                                "longitude": route_coordinates[i + 1][1],
                                "latitude": route_coordinates[i + 1][0]
                            }
                        }
                        road_segments.append(segment)
                
                clusters[f"cluster_{cluster_id + 1}"] = {
                    "vehicle_number": vehicle_number,
                    "vehicle_id": vehicle_id,
                    "depot": {"longitude": depot_point[0], "latitude": depot_point[1]},
                    "cluster_outline": [
                        {"longitude": coord[0], "latitude": coord[1]} 
                        for coord in cluster_outline
                    ],
                    "road_segments": road_segments,
                    "starting_point": {
                        "longitude": route_coordinates[0][1] if route_coordinates else depot_point[0],
                        "latitude": route_coordinates[0][0] if route_coordinates else depot_point[1]
                    },
                    "ending_point": {
                        "longitude": route_coordinates[-1][1] if route_coordinates else depot_point[0],
                        "latitude": route_coordinates[-1][0] if route_coordinates else depot_point[1]
                    },
                    "house_count": len(cluster_buildings)
                }
        
        return JSONResponse(clusters)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting clusters: {str(e)}")

@app.get("/generate-map/{map_type}")
async def generate_map(map_type: str):
    """Generate and return map HTML."""
    if map_type not in ["route_map", "cluster_analysis"]:
        raise HTTPException(status_code=400, detail="Invalid map type")
    
    file_path = os.path.join("output", f"{map_type}.html")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Map not found. Please upload files first using /optimize-routes")
    
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

def generate_map_from_files(ward_file, roads_file, buildings_file):
    """Generate map from file paths."""
    # Load GeoJSON data using geopandas
    ward_gdf = gpd.read_file(ward_file)
    roads_gdf = gpd.read_file(roads_file)
    buildings_gdf = gpd.read_file(buildings_file)
    
    # Convert to WGS84 if needed
    if ward_gdf.crs != 'EPSG:4326':
        ward_gdf = ward_gdf.to_crs('EPSG:4326')
    if roads_gdf.crs != 'EPSG:4326':
        roads_gdf = roads_gdf.to_crs('EPSG:4326')
    if buildings_gdf.crs != 'EPSG:4326':
        buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
    
    # Clean data - keep only geometry and essential columns
    ward_clean = ward_gdf[['geometry']]
    roads_clean = roads_gdf[['geometry']]
    buildings_clean = buildings_gdf[['geometry']]
    
    # Get center coordinates from ward bounds
    bounds = ward_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add ward boundaries (dark blue)
    folium.GeoJson(
        json.loads(ward_clean.to_json()),
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'darkblue',
            'weight': 3,
            'fillOpacity': 0
        }
    ).add_to(m)
    
    # Add road networks (black)
    folium.GeoJson(
        json.loads(roads_clean.to_json()),
        style_function=lambda x: {
            'color': 'black',
            'weight': 2
        }
    ).add_to(m)
    
    # Use K-means clustering on building centroids to ensure 5 clusters
    building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    building_clusters = kmeans.fit_predict(building_centroids)
    
    # Create road network graph
    G = nx.Graph()
    
    # Build road network for routing
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
    
    # Colors, vehicle names, and depot locations
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
    
    # Define depot/starting points for each vehicle (can be customized)
    depot_locations = [
        (center_lon - 0.002, center_lat - 0.002),  # Vehicle A depot
        (center_lon + 0.002, center_lat - 0.002),  # Vehicle B depot
        (center_lon, center_lat),                   # Vehicle C depot (center)
        (center_lon - 0.002, center_lat + 0.002),  # Vehicle D depot
        (center_lon + 0.002, center_lat + 0.002)   # Vehicle E depot
    ]
    
    # Process each cluster
    for cluster_id in range(5):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        
        if not cluster_buildings:
            continue
            
        # Find all road points for routing
        cluster_road_points = []
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    cluster_road_points.extend(list(line.coords))
            else:
                cluster_road_points.extend(list(geom.coords))
        
        # Create waste collection route covering all houses
        if cluster_buildings:
            cluster_road_points = list(set(cluster_road_points))
            
            # Get house locations for this cluster
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
            
            # Create collection route through all houses
            if house_road_points and len(house_road_points) > 0:
                # Remove duplicates while preserving order
                unique_house_points = []
                for pt in house_road_points:
                    if pt not in unique_house_points:
                        unique_house_points.append(pt)
                
                if len(unique_house_points) >= 1:
                    # Start from vehicle depot
                    depot_point = depot_locations[cluster_id]
                    start_point = unique_house_points[0]
                    
                    # Find nearest road point to depot
                    depot_distances = [((depot_point[0]-rp[0])**2 + (depot_point[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                    depot_road_point = cluster_road_points[np.argmin(depot_distances)]
                    
                    # Route: Depot -> Houses -> Depot
                    route_points = [depot_road_point]
                    
                    # Go to first house
                    try:
                        path_to_first = nx.shortest_path(G, depot_road_point, start_point, weight='weight')
                        route_points.extend(path_to_first[1:])
                    except:
                        route_points.append(start_point)
                    
                    # Visit all other houses
                    current_point = start_point
                    remaining_points = unique_house_points[1:].copy()
                    
                    while remaining_points:
                        # Find nearest unvisited house
                        distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                        nearest_idx = np.argmin(distances)
                        next_point = remaining_points.pop(nearest_idx)
                        
                        # Find shortest path on roads between current and next point
                        try:
                            path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                            route_points.extend(path_segment[1:])  # Skip first point to avoid duplication
                        except:
                            # If no path found, add direct connection
                            route_points.append(next_point)
                        
                        current_point = next_point
                    
                    # Return to depot
                    try:
                        path_to_depot = nx.shortest_path(G, current_point, depot_road_point, weight='weight')
                        route_points.extend(path_to_depot[1:])
                    except:
                        route_points.append(depot_road_point)
                    
                    # Convert to lat/lon for folium
                    route_coords = [[pt[1], pt[0]] for pt in route_points]
                    
                    # Calculate route distance
                    route_distance = sum([((route_points[i][0]-route_points[i+1][0])**2 + (route_points[i][1]-route_points[i+1][1])**2)**0.5 for i in range(len(route_points)-1)])
                    
                    # Add collection route with direction arrows
                    folium.PolyLine(
                        route_coords,
                        color=colors[cluster_id],
                        weight=4,
                        opacity=0.8,
                        popup=f"🚛 {vehicle_names[cluster_id]}<br>🏠 Houses: {len(cluster_buildings)}<br>📏 Distance: {route_distance:.4f} units<br>🔄 Route: Depot → Houses → Depot"
                    ).add_to(m)
                    
                    # Add directional arrows along the route
                    for i in range(0, len(route_coords)-1, max(1, len(route_coords)//10)):
                        if i+1 < len(route_coords):
                            # Calculate arrow direction
                            lat1, lon1 = route_coords[i]
                            lat2, lon2 = route_coords[i+1]
                            
                            # Calculate bearing for arrow rotation
                            dlon = math.radians(lon2 - lon1)
                            dlat = math.radians(lat2 - lat1)
                            bearing = math.degrees(math.atan2(dlon, dlat))
                            
                            # Add arrow marker
                            folium.Marker(
                                [lat1, lon1],
                                icon=folium.DivIcon(
                                    html=f'<div style="transform: rotate({bearing}deg); color: {colors[cluster_id]}; font-size: 16px;">➤</div>',
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10)
                                )
                            ).add_to(m)
                    
                    # Add depot start marker
                    folium.Marker(
                        [depot_road_point[1], depot_road_point[0]],
                        popup=f"{vehicle_names[cluster_id]} Depot (Start/End)<br>Route: {len(cluster_buildings)} houses<br>Distance: {len(route_points)} points",
                        icon=folium.Icon(color='darkgreen', icon='home')
                    ).add_to(m)
                    
                    # Add route info marker
                    folium.Marker(
                        [depot_point[1], depot_point[0]],
                        popup=f"🚛 {vehicle_names[cluster_id]}<br>📍 Depot Location<br>🏠 Houses: {len(cluster_buildings)}<br>📏 Route Points: {len(route_points)}",
                        icon=folium.Icon(color=colors[cluster_id], icon='truck')
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
            building = buildings_clean.iloc[building_idx]
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
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 300px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>🚛 Vehicle Route Assignments</h4>
    '''
    
    for i, (vehicle, color) in enumerate(zip(vehicle_names, colors)):
        cluster_count = len([j for j, c in enumerate(building_clusters) if c == i])
        legend_html += f'<p><span style="color:{color};">●</span> {vehicle}: {cluster_count} houses</p>'
    
    legend_html += '''
    <p><strong>🏠 Depot:</strong> Start/End point</p>
    <p><strong>➤</strong> Route direction</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m._repr_html_()


@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <body>
            <h2>Geospatial AI Route Optimizer</h2>
            <p>1. Upload files using POST /optimize-routes</p>
            <p>2. View generated map at /generate-map/route_map</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)