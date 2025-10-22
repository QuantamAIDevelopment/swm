"""FastAPI integration for geospatial route optimization."""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import tempfile
import os
import shutil
import geopandas as gpd
import pandas as pd
import folium
from sklearn.cluster import KMeans
import json
import networkx as nx
import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import math
from services.vehicle_service import VehicleService
from api.vehicles_api import router as vehicles_router
from loguru import logger

# API Key for authentication - Change this in production!
API_KEY = "swm-2024-secure-key"

# Security scheme
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header."""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

app = FastAPI(
    title="Geospatial AI Route Optimizer",
    description="Dynamic garbage collection route optimization using live vehicle data and road network",
    version="2.0.0"
)

# Initialize vehicle service
vehicle_service = VehicleService()

# Include vehicle API routes
app.include_router(vehicles_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_argmin(distances):
    """Safely get argmin, handling empty sequences."""
    if not distances or len(distances) == 0:
        return None
    return np.argmin(distances)

@app.post("/optimize-routes")
async def optimize_routes(
    roads_file: UploadFile = File(..., description="Roads GeoJSON file"),
    buildings_file: UploadFile = File(..., description="Buildings GeoJSON file"), 
    ward_geojson: UploadFile = File(..., description="Ward boundary GeoJSON file"),
    ward_no: str = Form(..., description="Ward number to filter vehicles")
):
    """Upload files and run complete route optimization pipeline."""
    
    # Validate file types and ward_no
    if not roads_file.filename or not roads_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Roads file must be GeoJSON")
    if not buildings_file.filename or not buildings_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Buildings file must be GeoJSON")
    if not ward_no or not ward_no.strip():
        raise HTTPException(status_code=400, detail="Ward number is required")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded files
            roads_path = os.path.join(temp_dir, "roads.geojson")
            buildings_path = os.path.join(temp_dir, "buildings.geojson")
            ward_path = os.path.join(temp_dir, "ward.geojson")
            
            with open(roads_path, "wb") as f:
                shutil.copyfileobj(roads_file.file, f)
            with open(buildings_path, "wb") as f:
                shutil.copyfileobj(buildings_file.file, f)
            with open(ward_path, "wb") as f:
                shutil.copyfileobj(ward_geojson.file, f)
            
            # Get live vehicle data filtered by ward
            try:
                vehicles_df = vehicle_service.get_vehicles_by_ward(ward_no.strip())
                print(f"Loaded {len(vehicles_df)} vehicles for ward {ward_no}")
                
                if len(vehicles_df) == 0:
                    raise HTTPException(status_code=404, detail=f"No active vehicles found for ward {ward_no}")
                
                # Save vehicle data for map generation
                vehicles_csv_path = os.path.join(temp_dir, "vehicles.csv")
                vehicles_df.to_csv(vehicles_csv_path, index=False)
                vehicles_path = vehicles_csv_path
                
            except HTTPException:
                raise
            except Exception as vehicle_error:
                print(f"Failed to get live vehicle data: {vehicle_error}")
                raise HTTPException(status_code=500, detail="Failed to get vehicle data from API")
            
            # Generate map using uploaded files
            try:
                map_html = generate_map_from_files(ward_path, roads_path, buildings_path, vehicles_path)
                print("Map generation completed successfully")
            except Exception as map_error:
                print(f"Map generation error: {map_error}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                # Create simple fallback map
                map_html = "<html><body><h1>Map Processing Complete</h1><p>Data uploaded successfully</p></body></html>"
            
            # Save map and data to output directory
            os.makedirs("output", exist_ok=True)
            try:
                with open("output/route_map.html", "w", encoding="utf-8") as f:
                    f.write(map_html)
            except Exception as save_error:
                print(f"File save error: {save_error}")
                raise save_error
            
            # Save data files for cluster endpoint
            shutil.copy(ward_path, "output/ward.geojson")
            shutil.copy(buildings_path, "output/buildings.geojson")
            shutil.copy(roads_path, "output/roads.geojson")
            shutil.copy(vehicles_path, "output/vehicles.csv")
            
            # Prepare vehicle data for response
            vehicle_data = []
            for _, vehicle in vehicles_df.iterrows():
                vehicle_data.append({
                    "vehicle_id": str(vehicle.get('vehicle_id', 'N/A')),
                    "vehicle_type": str(vehicle.get('vehicle_type', 'N/A')),
                    "status": str(vehicle.get('status', 'N/A')),
                    "ward_no": str(vehicle.get('ward_no', 'N/A')),
                    "driver_name": str(vehicle.get('driverName', 'N/A')),
                    "capacity": int(vehicle.get('capacity', 0)) if pd.notna(vehicle.get('capacity', 0)) else 0
                })
            
            return JSONResponse({
                "status": "success",
                "message": f"Route optimization completed for ward {ward_no} with {len(vehicles_df)} live vehicles",
                "maps": {
                    "route_map": "/generate-map/route_map",
                    "cluster_analysis": "/generate-map/cluster_analysis"
                },
                "dashboard": "/cluster-dashboard",
                "ward_no": ward_no,
                "vehicle_count": len(vehicles_df),
                "vehicle_source": "Live API (Ward Filtered)",
                "vehicles": vehicle_data,
                "features": [
                    "Live vehicle data integration",
                    "Ward-based vehicle filtering",
                    "Interactive cluster dashboard",
                    "Real vehicle information in maps",
                    "Color-coded routes and buildings"
                ]
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/cluster/{cluster_id}")
async def get_cluster(cluster_id: int, api_key: str = Depends(verify_api_key)):
    """Return detailed data for a specific cluster."""
    try:
        # Check if files exist
        ward_path = "output/ward.geojson"
        buildings_path = "output/buildings.geojson"
        roads_path = "output/roads.geojson"
        
        if not os.path.exists(ward_path) or not os.path.exists(buildings_path):
            raise HTTPException(status_code=404, detail="Data not found. Please upload files first using /optimize-routes")
        
        # Validate cluster_id
        if cluster_id < 1 or cluster_id > 5:
            raise HTTPException(status_code=400, detail="Cluster ID must be between 1 and 5")
        
        # Load data
        buildings_gdf = gpd.read_file(buildings_path)
        roads_gdf = gpd.read_file(roads_path) if os.path.exists(roads_path) else None
        
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        if roads_gdf is not None and roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        
        # Cluster buildings
        building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        n_clusters = min(5, len(building_centroids))
        
        if n_clusters == 1:
            building_clusters = [0] * len(building_centroids)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            building_clusters = kmeans.fit_predict(building_centroids)
        
        # Get center coordinates for depot calculation
        bounds = buildings_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Define depot locations
        depot_locations = [
            (center_lon - 0.002, center_lat - 0.002),
            (center_lon + 0.002, center_lat - 0.002),
            (center_lon, center_lat),
            (center_lon - 0.002, center_lat + 0.002),
            (center_lon + 0.002, center_lat + 0.002)
        ]
        
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
        
        # Create road network graph
        G = nx.Graph()
        all_road_points = []
        
        if roads_gdf is not None:
            for idx, road in roads_gdf.iterrows():
                geom = road.geometry
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords = list(line.coords)
                        all_road_points.extend(coords)
                        for i in range(len(coords)-1):
                            p1, p2 = coords[i], coords[i+1]
                            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                            G.add_edge(p1, p2, weight=dist)
                else:
                    coords = list(geom.coords)
                    all_road_points.extend(coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
        
        all_road_points = list(set(all_road_points))
        
        # Process the specific cluster
        target_cluster_id = cluster_id - 1  # Convert to 0-based index
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == target_cluster_id]
        
        if not cluster_buildings:
            raise HTTPException(status_code=404, detail=f"No buildings found in cluster {cluster_id}")
        
        cluster_coords = [building_centroids[i] for i in cluster_buildings]
        depot = depot_locations[target_cluster_id] if target_cluster_id < len(depot_locations) else depot_locations[0]
        
        # Get house locations for this cluster
        house_locations_wgs84 = []
        for i in cluster_buildings:
            pt = buildings_gdf.iloc[i].geometry.centroid
            house_locations_wgs84.append((pt.x, pt.y))
        
        # Find nearest road points to houses and depot
        house_road_points = []
        for house_pt in house_locations_wgs84:
            if all_road_points:
                distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in all_road_points]
                nearest_idx = np.argmin(distances)
                house_road_points.append(all_road_points[nearest_idx])
            else:
                house_road_points.append(house_pt)
        
        # Find depot road point
        if all_road_points:
            depot_distances = [((depot[0]-rp[0])**2 + (depot[1]-rp[1])**2)**0.5 for rp in all_road_points]
            depot_road_point = all_road_points[np.argmin(depot_distances)]
        else:
            depot_road_point = depot
        
        # Create optimized route
        route_points = [depot_road_point]  # Start at depot
        road_segments = []
        
        if house_road_points:
            # Remove duplicates while preserving order
            unique_house_points = []
            for pt in house_road_points:
                if pt not in unique_house_points:
                    unique_house_points.append(pt)
            
            if unique_house_points:
                current_point = depot_road_point
                remaining_points = unique_house_points.copy()
                
                # Visit all houses using shortest path
                while remaining_points:
                    # Find nearest unvisited house
                    distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                    nearest_idx = np.argmin(distances)
                    next_point = remaining_points.pop(nearest_idx)
                    
                    # Try to find path on road network
                    try:
                        path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                        route_points.extend(path_segment[1:])  # Skip first point
                        
                        # Add road segment details
                        for i in range(len(path_segment)-1):
                            road_segments.append({
                                "start": {"longitude": path_segment[i][0], "latitude": path_segment[i][1]},
                                "end": {"longitude": path_segment[i+1][0], "latitude": path_segment[i+1][1]}
                            })
                    except:
                        # Direct connection if no path found
                        route_points.append(next_point)
                        road_segments.append({
                            "start": {"longitude": current_point[0], "latitude": current_point[1]},
                            "end": {"longitude": next_point[0], "latitude": next_point[1]}
                        })
                    
                    current_point = next_point
                
                # Return to depot
                try:
                    path_home = nx.shortest_path(G, current_point, depot_road_point, weight='weight')
                    route_points.extend(path_home[1:])
                    
                    # Add return segments
                    for i in range(len(path_home)-1):
                        road_segments.append({
                            "start": {"longitude": path_home[i][0], "latitude": path_home[i][1]},
                            "end": {"longitude": path_home[i+1][0], "latitude": path_home[i+1][1]}
                        })
                except:
                    route_points.append(depot_road_point)
                    road_segments.append({
                        "start": {"longitude": current_point[0], "latitude": current_point[1]},
                        "end": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]}
                    })
        
        # Create cluster outline (convex hull)
        from shapely.geometry import MultiPoint
        cluster_points = MultiPoint(cluster_coords + [depot])
        cluster_outline = list(cluster_points.convex_hull.exterior.coords)
        
        cluster_data = {
            "cluster_id": cluster_id,
            "vehicle": vehicle_names[target_cluster_id] if target_cluster_id < len(vehicle_names) else f"Vehicle {cluster_id}",
            "vehicle_start_point": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]},
            "vehicle_end_point": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]},
            "depot": {"longitude": depot[0], "latitude": depot[1]},
            "cluster_outline": [
                {"longitude": coord[0], "latitude": coord[1]} 
                for coord in cluster_outline
            ],
            "road_segments": road_segments,
            "complete_route_coordinates": [
                {"longitude": pt[0], "latitude": pt[1]} 
                for pt in route_points
            ],
            "house_count": len(cluster_coords),
            "total_segments": len(road_segments)
        }
        
        return JSONResponse(cluster_data)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Cluster error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error getting cluster {cluster_id}: {str(e)}")

@app.get("/cluster-dashboard")
async def get_cluster_dashboard(api_key: str = Depends(verify_api_key)):
    """Return cluster dashboard data for visualization."""
    try:
        buildings_path = "output/buildings.geojson"
        if not os.path.exists(buildings_path):
            raise HTTPException(status_code=404, detail="Data not found. Upload files first.")
        
        buildings_gdf = gpd.read_file(buildings_path)
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        
        building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        n_clusters = min(5, len(building_centroids))
        
        if n_clusters == 1:
            building_clusters = [0] * len(building_centroids)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            building_clusters = kmeans.fit_predict(building_centroids)
        
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF']
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
        
        dashboard_data = []
        for cluster_id in range(n_clusters):
            cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
            if cluster_buildings:
                dashboard_data.append({
                    "cluster_id": cluster_id + 1,
                    "color": colors[cluster_id],
                    "vehicle": vehicle_names[cluster_id],
                    "building_count": len(cluster_buildings),
                    "estimated_distance": f"{len(cluster_buildings) * 0.5:.1f} km",
                    "estimated_time": f"{len(cluster_buildings) * 3:.0f} min"
                })
        
        return JSONResponse({
            "total_clusters": n_clusters,
            "total_buildings": len(buildings_gdf),
            "clusters": dashboard_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/clusters")
async def get_all_clusters(api_key: str = Depends(verify_api_key)):
    """Return all clusters data."""
    clusters = {}
    for i in range(1, 6):
        try:
            cluster_response = await get_cluster(i)
            cluster_data = json.loads(cluster_response.body)
            clusters[f"cluster_{i}"] = cluster_data
        except:
            continue
    try:
        # Check if files exist
        ward_path = "output/ward.geojson"
        buildings_path = "output/buildings.geojson"
        roads_path = "output/roads.geojson"
        
        if not os.path.exists(ward_path) or not os.path.exists(buildings_path):
            raise HTTPException(status_code=404, detail="Data not found. Please upload files first using /optimize-routes")
        
        # Load data
        buildings_gdf = gpd.read_file(buildings_path)
        roads_gdf = gpd.read_file(roads_path) if os.path.exists(roads_path) else None
        
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        if roads_gdf is not None and roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        
        # Validate we have buildings
        if len(buildings_gdf) == 0:
            raise HTTPException(status_code=400, detail="No buildings found in the dataset")
        
        # Cluster buildings
        building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        
        if len(building_centroids) == 0:
            raise HTTPException(status_code=400, detail="No valid building centroids found")
        
        n_clusters = min(5, len(building_centroids))
        
        if n_clusters == 1:
            building_clusters = [0] * len(building_centroids)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            building_clusters = kmeans.fit_predict(building_centroids)
        
        # Get center coordinates for depot calculation
        bounds = buildings_gdf.total_bounds
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
        
        # Create road network graph
        G = nx.Graph()
        all_road_points = []
        
        if roads_gdf is not None:
            for idx, road in roads_gdf.iterrows():
                geom = road.geometry
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coords = list(line.coords)
                        all_road_points.extend(coords)
                        for i in range(len(coords)-1):
                            p1, p2 = coords[i], coords[i+1]
                            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                            G.add_edge(p1, p2, weight=dist)
                else:
                    coords = list(geom.coords)
                    all_road_points.extend(coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
        
        all_road_points = list(set(all_road_points))
        
        # Create detailed cluster response
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
            
            if cluster_buildings:
                cluster_coords = [building_centroids[i] for i in cluster_buildings]
                depot = depot_locations[cluster_id] if cluster_id < len(depot_locations) else depot_locations[0]
                
                # Get house locations for this cluster
                house_locations_wgs84 = []
                for i in cluster_buildings:
                    pt = buildings_gdf.iloc[i].geometry.centroid
                    house_locations_wgs84.append((pt.x, pt.y))
                
                # Find nearest road points to houses and depot
                house_road_points = []
                for house_pt in house_locations_wgs84:
                    if all_road_points:
                        distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in all_road_points]
                        nearest_idx = np.argmin(distances)
                        house_road_points.append(all_road_points[nearest_idx])
                    else:
                        house_road_points.append(house_pt)
                
                # Find depot road point
                if all_road_points:
                    depot_distances = [((depot[0]-rp[0])**2 + (depot[1]-rp[1])**2)**0.5 for rp in all_road_points]
                    depot_road_point = all_road_points[np.argmin(depot_distances)]
                else:
                    depot_road_point = depot
                
                # Create optimized route
                route_points = [depot_road_point]  # Start at depot
                road_segments = []
                
                if house_road_points:
                    # Remove duplicates while preserving order
                    unique_house_points = []
                    for pt in house_road_points:
                        if pt not in unique_house_points:
                            unique_house_points.append(pt)
                    
                    if unique_house_points:
                        current_point = depot_road_point
                        remaining_points = unique_house_points.copy()
                        
                        # Visit all houses using shortest path
                        while remaining_points:
                            # Find nearest unvisited house
                            distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                            nearest_idx = np.argmin(distances)
                            next_point = remaining_points.pop(nearest_idx)
                            
                            # Try to find path on road network
                            try:
                                path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                                route_points.extend(path_segment[1:])  # Skip first point
                                
                                # Add road segment details
                                for i in range(len(path_segment)-1):
                                    road_segments.append({
                                        "start": {"longitude": path_segment[i][0], "latitude": path_segment[i][1]},
                                        "end": {"longitude": path_segment[i+1][0], "latitude": path_segment[i+1][1]}
                                    })
                            except:
                                # Direct connection if no path found
                                route_points.append(next_point)
                                road_segments.append({
                                    "start": {"longitude": current_point[0], "latitude": current_point[1]},
                                    "end": {"longitude": next_point[0], "latitude": next_point[1]}
                                })
                            
                            current_point = next_point
                        
                        # Return to depot
                        try:
                            path_home = nx.shortest_path(G, current_point, depot_road_point, weight='weight')
                            route_points.extend(path_home[1:])
                            
                            # Add return segments
                            for i in range(len(path_home)-1):
                                road_segments.append({
                                    "start": {"longitude": path_home[i][0], "latitude": path_home[i][1]},
                                    "end": {"longitude": path_home[i+1][0], "latitude": path_home[i+1][1]}
                                })
                        except:
                            route_points.append(depot_road_point)
                            road_segments.append({
                                "start": {"longitude": current_point[0], "latitude": current_point[1]},
                                "end": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]}
                            })
                
                # Create cluster outline (convex hull)
                from shapely.geometry import MultiPoint
                cluster_points = MultiPoint(cluster_coords + [depot])
                cluster_outline = list(cluster_points.convex_hull.exterior.coords)
                
                clusters[f"cluster_{cluster_id + 1}"] = {
                    "vehicle": vehicle_names[cluster_id] if cluster_id < len(vehicle_names) else f"Vehicle {cluster_id + 1}",
                    "vehicle_start_point": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]},
                    "vehicle_end_point": {"longitude": depot_road_point[0], "latitude": depot_road_point[1]},
                    "depot": {"longitude": depot[0], "latitude": depot[1]},
                    "cluster_outline": [
                        {"longitude": coord[0], "latitude": coord[1]} 
                        for coord in cluster_outline
                    ],
                    "road_segments": road_segments,
                    "complete_route_coordinates": [
                        {"longitude": pt[0], "latitude": pt[1]} 
                        for pt in route_points
                    ],
                    "house_count": len(cluster_coords),
                    "total_segments": len(road_segments)
                }
        
        return JSONResponse(clusters)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Cluster error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error getting clusters: {str(e)}")

@app.get("/generate-map/{map_type}")
async def generate_map(map_type: str):
    """Generate and return map HTML."""
    # Allow any map type
    pass
    
    file_path = os.path.join("output", f"{map_type}.html")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Map not found. Please upload files first using /optimize-routes")
    
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

def generate_map_from_files(ward_file, roads_file, buildings_file, vehicles_file=None):
    """Generate map with layer controls."""
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
    buildings_clean = buildings_gdf[['geometry']]
    
    # Get center coordinates from ward bounds
    bounds = ward_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add ward boundaries to base layer
    ward_layer = folium.FeatureGroup(name="Ward Boundary", show=True)
    folium.GeoJson(
        json.loads(ward_clean.to_json()),
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'darkblue',
            'weight': 3,
            'fillOpacity': 0
        }
    ).add_to(ward_layer)
    ward_layer.add_to(m)
    
    # Load vehicle data if available
    vehicles_df = None
    if vehicles_file and os.path.exists(vehicles_file):
        import pandas as pd
        vehicles_df = pd.read_csv(vehicles_file)
        print(f"Loaded {len(vehicles_df)} vehicles for map generation")
    
    # Use vehicle count for clustering or default to 5
    n_vehicles = len(vehicles_df) if vehicles_df is not None else 5
    building_centroids = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
    kmeans = KMeans(n_clusters=min(n_vehicles, len(building_centroids)), random_state=42, n_init=10)
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
    
    # Colors and vehicle names from live data or defaults
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if vehicles_df is not None:
        vehicle_names = vehicles_df['vehicle_id'].tolist()[:len(set(building_clusters))]
    else:
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
    
    # Process each cluster with separate layers
    n_clusters = len(set(building_clusters))
    for cluster_id in range(n_clusters):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        
        if not cluster_buildings:
            continue
        
        # Create separate layer for each cluster with vehicle info
        vehicle_name = vehicle_names[cluster_id] if cluster_id < len(vehicle_names) else f"Vehicle {cluster_id + 1}"
        vehicle_info = ""
        if vehicles_df is not None and cluster_id < len(vehicles_df):
            vehicle = vehicles_df.iloc[cluster_id]
            vehicle_info = f" ({vehicle.get('vehicle_type', 'N/A')} - {vehicle.get('status', 'N/A')})"
        
        cluster_layer = folium.FeatureGroup(
            name=f"🚛 {vehicle_name}{vehicle_info} - {len(cluster_buildings)} buildings",
            show=True
        )
            
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
                if cluster_road_points:
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
                    # Start from first house location
                    start_point = unique_house_points[0]
                    route_points = [start_point]
                    
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
                    
                    # Convert to lat/lon for folium
                    route_coords = [[pt[1], pt[0]] for pt in route_points]
                    
                    # Add collection route with direction arrows
                    route_popup = f"{vehicle_name} - {len(cluster_buildings)} Houses"
                    if vehicles_df is not None and cluster_id < len(vehicles_df):
                        vehicle = vehicles_df.iloc[cluster_id]
                        route_popup += f"\nType: {vehicle.get('vehicle_type', 'N/A')}\nDriver: {vehicle.get('driverName', 'N/A')}"
                    
                    folium.PolyLine(
                        route_coords,
                        color=colors[cluster_id % len(colors)],
                        weight=4,
                        opacity=0.8,
                        popup=route_popup
                    ).add_to(cluster_layer)
                    
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
                                    html=f'<div style="transform: rotate({bearing}deg); color: {colors[cluster_id % len(colors)]}; font-size: 16px;">➤</div>',
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10)
                                )
                            ).add_to(cluster_layer)
                    
                    # Add start marker with vehicle info
                    start_popup = f"{vehicle_name} Start"
                    if vehicles_df is not None and cluster_id < len(vehicles_df):
                        vehicle = vehicles_df.iloc[cluster_id]
                        start_popup += f"\nID: {vehicle.get('vehicle_id', 'N/A')}\nWard: {vehicle.get('ward_no', 'N/A')}"
                    
                    folium.Marker(
                        [start_point[1], start_point[0]],
                        popup=start_popup,
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(cluster_layer)
                    
                    # Add end marker
                    folium.Marker(
                        [current_point[1], current_point[0]],
                        popup=f"{vehicle_name} End",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(cluster_layer)
        
        # Add clustered buildings as polygons
        for house_number, building_idx in enumerate(cluster_buildings, 1):
            building = buildings_clean.iloc[building_idx]
            folium.GeoJson(
                json.loads(gpd.GeoSeries([building.geometry]).to_json()),
                style_function=lambda x, c=colors[cluster_id % len(colors)]: {
                    'fillColor': c,
                    'color': c,
                    'weight': 1,
                    'fillOpacity': 0.6
                },
                popup=f"{vehicle_name} - House {house_number}",
                tooltip=f"C{cluster_id + 1}-H{house_number}"
            ).add_to(cluster_layer)
        
        # Add cluster layer to map
        cluster_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)
    
    # Add cluster dashboard panel with layer toggle functionality
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        if cluster_buildings:
            vehicle_name = vehicle_names[cluster_id] if cluster_id < len(vehicle_names) else f"Vehicle {cluster_id + 1}"
            vehicle_details = ""
            if vehicles_df is not None and cluster_id < len(vehicles_df):
                vehicle = vehicles_df.iloc[cluster_id]
                vehicle_details = f" • {vehicle.get('vehicle_type', 'N/A')} • Ward {vehicle.get('ward_no', 'N/A')}"
            
            cluster_stats.append(f'''
            <div style="margin:5px 0;padding:8px;border:1px solid #ddd;border-radius:4px;background:#f9f9f9;">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div>
                        <span style="color:{colors[cluster_id % len(colors)]};font-size:14px;">●</span> 
                        <strong>Cluster {cluster_id + 1}</strong>
                    </div>
                    <button onclick="toggleCluster({cluster_id})" style="padding:2px 6px;font-size:10px;border:1px solid {colors[cluster_id % len(colors)]};background:white;border-radius:3px;cursor:pointer;">Toggle</button>
                </div>
                <div style="font-size:11px;margin-top:5px;">
                    {len(cluster_buildings)} buildings<br>
                    <small>{vehicle_name}{vehicle_details} • {len(cluster_buildings) * 0.5:.1f}km • {len(cluster_buildings) * 3:.0f}min</small>
                </div>
            </div>
            ''')
    
    panel_html = f'''
    <div style="position:fixed;top:10px;right:10px;width:280px;max-height:70vh;background:white;border:2px solid #333;z-index:9999;font-size:12px;border-radius:5px;box-shadow:0 2px 10px rgba(0,0,0,0.3);">
        <div style="background:#333;color:white;padding:8px;border-radius:3px 3px 0 0;">
            <strong>📊 Cluster Dashboard</strong>
            <div style="font-size:10px;margin-top:3px;">{len([c for c in cluster_stats if c])} clusters • {len(buildings_gdf)} buildings</div>
            <div style="margin-top:5px;">
                <button onclick="showAllClusters()" style="padding:3px 8px;font-size:10px;border:1px solid white;background:none;color:white;border-radius:3px;cursor:pointer;margin-right:5px;">Show All</button>
                <button onclick="hideAllClusters()" style="padding:3px 8px;font-size:10px;border:1px solid white;background:none;color:white;border-radius:3px;cursor:pointer;">Hide All</button>
            </div>
        </div>
        <div style="padding:8px;max-height:50vh;overflow-y:auto;">
            {''.join(cluster_stats)}
        </div>
    </div>
    
    <script>
    function toggleCluster(clusterId) {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Cluster ' + (clusterId + 1))) {{
                control.click();
            }}
        }});
    }}
    
    function showAllClusters() {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Cluster') && !control.checked) {{
                control.click();
            }}
        }});
    }}
    
    function hideAllClusters() {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Cluster') && control.checked) {{
                control.click();
            }}
        }});
    }}
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(panel_html))
    
    return m._repr_html_()

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <body>
            <h2>🗺️ Geospatial AI Route Optimizer</h2>
            <div style="background:#fff3cd;border:1px solid #ffeaa7;padding:10px;margin:10px 0;border-radius:5px;">
                <strong>🔐 API Key Required:</strong> <code>swm-2024-secure-key</code><br>
                <small>Add to Authorization header: <code>Bearer swm-2024-secure-key</code></small>
            </div>
            <h3>Available Endpoints:</h3>
            <ul>
                <li><strong>POST /optimize-routes</strong> - Upload files with ward_no and generate optimized routes using live vehicles</li>
                <li><strong>GET /generate-map/route_map</strong> - View interactive map with layer controls</li>
                <li><strong>GET /cluster-dashboard</strong> - Get cluster data in JSON format</li>
                <li><strong>GET /cluster/{cluster_id}</strong> - Get specific cluster details</li>
                <li><strong>GET /clusters</strong> - Get all clusters data</li>
                <li><strong>GET /api/vehicles/live</strong> - Get live vehicle data from SWM API</li>
                <li><strong>GET /api/vehicles/{vehicle_id}</strong> - Get specific vehicle details</li>
                <li><strong>PUT /api/vehicles/{vehicle_id}/status</strong> - Update vehicle status</li>
            </ul>
            <h3>Features:</h3>
            <ul>
                <li>🌐 <strong>Ward-based Vehicle Filtering</strong> - Real-time vehicle data filtered by ward number</li>
                <li>✅ Interactive cluster dashboard panel</li>
                <li>✅ Layer controls to show/hide individual clusters</li>
                <li>✅ Toggle buttons for each cluster</li>
                <li>✅ Show All / Hide All cluster controls</li>
                <li>✅ Color-coded routes and buildings</li>
                <li>🔐 API Key authentication</li>
                <li>📱 RESTful vehicle management endpoints</li>
                <li>🏘️ Ward-based vehicle clustering and optimization</li>
            </ul>
            <p><a href="/docs" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">📚 API Documentation</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)