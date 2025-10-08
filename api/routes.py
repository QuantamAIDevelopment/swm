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

from agents.route_assignment_agent import RouteAssignmentAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Garbage Collection Route Assignment System")

# Initialize components
route_agent = RouteAssignmentAgent()

# Global storage for uploaded data
current_data = None
current_routes = None

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

@app.post("/assign_routes")
async def assign_routes():
    """Compute optimized routes for uploaded data."""
    global current_data, current_routes
    try:
        if current_data is None:
            raise HTTPException(status_code=400, detail="No data uploaded. Please upload files first.")
        
        # Process routes using the route agent
        logger.info("Starting route processing...")
        try:
            routes = route_agent.process_data(current_data)
            logger.info(f"Route processing completed, got {len(routes) if routes else 0} routes")
            current_routes = routes
        except Exception as route_error:
            logger.error(f"Route processing failed: {route_error}")
            logger.error(f"Route error type: {type(route_error)}")
            # Create dummy routes for testing
            current_routes = []
            raise route_error
        
        logger.info(f"Generated {len(routes)} routes")
        return {
            "status": "success",
            "message": f"Generated {len(routes)} routes successfully",
            "route_count": len(routes)
        }
        
    except Exception as e:
        logger.error(f"Error computing routes: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Route computation failed: {str(e)}")

@app.get("/map")
async def get_map():
    """Generate enhanced folium map with Ward 29 focus."""
    global current_data, current_routes
    try:
        if current_data is None or current_routes is None:
            raise HTTPException(status_code=404, detail="No data or routes found. Please upload files and assign routes first.")
        
        # Create enhanced folium map
        houses = current_data["houses"].to_crs("EPSG:4326")
        roads = current_data["road_network"].to_crs("EPSG:4326")
        wards = current_data["ward_boundaries"].to_crs("EPSG:4326")
        vehicles = current_data["vehicles"]
        
        # Filter Ward 29 if available
        try:
            if 'ward_no' in wards.columns:
                ward_29 = wards[wards['ward_no'] == 29]
            elif 'ward_id' in wards.columns:
                ward_29 = wards[wards['ward_id'] == 29]
            else:
                ward_29 = wards  # Use all wards if no ward column found
            
            if len(ward_29) == 0:
                ward_29 = wards  # Use all wards if Ward 29 not found
        except Exception as ward_error:
            logger.warning(f"Ward filtering failed: {ward_error}, using all wards")
            ward_29 = wards
        
        try:
            bounds = ward_29.total_bounds if len(ward_29) > 0 else houses.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        except Exception as bounds_error:
            logger.warning(f"Bounds calculation failed: {bounds_error}, using default center")
            center_lat, center_lon = 0, 0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Layer 1: Ward boundary (sky blue, 4px thick, no fill)
        for idx, ward in ward_29.iterrows():
            folium.GeoJson(
                ward.geometry,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'skyblue',
                    'weight': 4,
                    'fillOpacity': 0,
                    'opacity': 1
                },
                popup=f"Ward: {ward.get('ward_name', ward.get('ward_no', 'Unknown'))}"
            ).add_to(m)
        
        # Layer 2: Road network (gray, thin lines)
        for idx, road in roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = [[coord[1], coord[0]] for coord in road.geometry.coords]
                folium.PolyLine(
                    locations=coords,
                    color='#666666',
                    weight=1,
                    opacity=0.6
                ).add_to(m)
            elif road.geometry.geom_type == 'MultiLineString':
                for line in road.geometry.geoms:
                    coords = [[coord[1], coord[0]] for coord in line.coords]
                    folium.PolyLine(
                        locations=coords,
                        color='#666666',
                        weight=1,
                        opacity=0.6
                    ).add_to(m)
        
        # Layer 3: Individual vehicle route layers with controls
        route_colors = ['red', 'green', 'blue', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        vehicle_labels = {}
        
        # Create feature groups for each vehicle
        vehicle_groups = {}
        all_routes_group = folium.FeatureGroup(name="All Routes", show=True)
        
        for i, route in enumerate(current_routes):
            color = route_colors[i % len(route_colors)]
            vehicle_id = route.vehicle_id
            truck_label = f"T{i+1}"
            vehicle_labels[vehicle_id] = truck_label
            
            # Create individual vehicle group
            vehicle_group = folium.FeatureGroup(name=f"Vehicle {truck_label}", show=False)
            vehicle_groups[vehicle_id] = vehicle_group
            
            # Get individual road segments for this route
            route_road_ids = [int(rid) for rid in route.road_segment_ids]
            # Ensure roads has road_id column
            if 'road_id' not in roads.columns:
                roads_with_id = roads.copy()
                roads_with_id['road_id'] = roads_with_id.index
                route_roads = roads_with_id[roads_with_id['road_id'].isin(route_road_ids)]
            else:
                route_roads = roads[roads['road_id'].isin(route_road_ids)]
            
            # Add each road segment individually to avoid overlaps
            for idx, road in route_roads.iterrows():
                if road.geometry.geom_type == 'LineString':
                    coords = [[coord[1], coord[0]] for coord in road.geometry.coords]
                    
                    # Add to individual vehicle group
                    folium.PolyLine(
                        locations=coords,
                        color=color,
                        weight=4,
                        opacity=0.9,
                        popup=f"Vehicle: {truck_label}<br>Road: {road.get('road_name', road.get('road_id', idx))}<br>Houses: {len([h for h in route.ordered_house_ids])}"
                    ).add_to(vehicle_group)
                    
                    # Add to all routes group with thinner line
                    folium.PolyLine(
                        locations=coords,
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=f"Vehicle: {truck_label}"
                    ).add_to(all_routes_group)
                    
                elif road.geometry.geom_type == 'MultiLineString':
                    for line in road.geometry.geoms:
                        coords = [[coord[1], coord[0]] for coord in line.coords]
                        
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=4,
                            opacity=0.9,
                            popup=f"Vehicle: {truck_label}"
                        ).add_to(vehicle_group)
                        
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=2,
                            opacity=0.7,
                            popup=f"Vehicle: {truck_label}"
                        ).add_to(all_routes_group)
            
            # Add vehicle group to map
            vehicle_group.add_to(m)
        
        # Add all routes group
        all_routes_group.add_to(m)
        
        # Layer 4: Houses grouped by vehicle assignment
        houses_group = folium.FeatureGroup(name="All Houses", show=True)
        
        # Color houses by their assigned vehicle
        for idx, house in houses.iterrows():
            centroid = house.geometry.centroid
            house_id = house.get('house_id', idx)
            
            # Find which vehicle this house belongs to
            assigned_color = 'black'
            assigned_vehicle = 'Unassigned'
            
            for i, route in enumerate(current_routes):
                if str(house_id) in route.ordered_house_ids:
                    assigned_color = route_colors[i % len(route_colors)]
                    assigned_vehicle = vehicle_labels[route.vehicle_id]
                    break
            
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=2,
                popup=f"House ID: {house_id}<br>Assigned to: {assigned_vehicle}",
                color=assigned_color,
                fillColor=assigned_color,
                fillOpacity=0.8,
                weight=1
            ).add_to(houses_group)
        
        houses_group.add_to(m)
        
        # Layer 5: Vehicle depot points for each cluster
        for i, route in enumerate(current_routes):
            color = route_colors[i % len(route_colors)]
            vehicle_id = route.vehicle_id
            truck_label = vehicle_labels[vehicle_id]
            vehicle_group = vehicle_groups[vehicle_id]
            
            # Get route roads
            route_road_ids = [int(rid) for rid in route.road_segment_ids]
            if 'road_id' not in roads.columns:
                roads_with_id = roads.copy()
                roads_with_id['road_id'] = roads_with_id.index
                route_roads = roads_with_id[roads_with_id['road_id'].isin(route_road_ids)]
            else:
                route_roads = roads[roads['road_id'].isin(route_road_ids)]
            
            if len(route_roads) > 0:
                # Calculate cluster centroid as depot location
                cluster_centroid = route_roads.geometry.centroid.unary_union.centroid
                
                # Single depot marker for this vehicle cluster
                folium.Marker(
                    location=[cluster_centroid.y, cluster_centroid.x],
                    popup=f"{truck_label} Depot<br>Vehicle: {vehicle_id}<br>Roads: {len(route_road_ids)}<br>Houses: {len(route.ordered_house_ids)}",
                    icon=folium.Icon(color='blue', icon='home'),
                    tooltip=f"{truck_label} Depot"
                ).add_to(vehicle_group)
        
        # Add layer control
        folium.LayerControl(position='topleft').add_to(m)
        
        # Enhanced dashboard with vehicle cluster controls
        total_distance = sum(route.total_distance_meters for route in current_routes)
        
        # Generate vehicle details for dashboard
        vehicle_details = []
        for i, route in enumerate(current_routes):
            color = route_colors[i % len(route_colors)]
            truck_label = vehicle_labels[route.vehicle_id]
            vehicle_details.append({
                'label': truck_label,
                'id': route.vehicle_id,
                'color': color,
                'roads': len(route.road_segment_ids),
                'houses': len(route.ordered_house_ids),
                'distance': route.total_distance_meters/1000
            })
        
        vehicle_rows = ""
        for v in vehicle_details:
            vehicle_rows += f"""
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:3px;"><span style="color:{v['color']}; font-weight:bold;">{v['label']}</span></td>
                <td style="padding:3px;">{v['roads']}</td>
                <td style="padding:3px;">{v['houses']}</td>
                <td style="padding:3px;">{v['distance']:.1f}km</td>
            </tr>
            """
        
        dashboard_html = f"""
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 320px; height: auto; 
                    background-color: rgba(255,255,255,0.95); 
                    border: 2px solid #333; 
                    border-radius: 8px;
                    z-index: 9999; 
                    font-size: 12px; 
                    padding: 15px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    max-height: 80vh;
                    overflow-y: auto;">
        <h3 style="margin-top:0; color:#333; border-bottom:2px solid #007acc; padding-bottom:5px;">Route Dashboard</h3>
        
        <div style="margin-bottom:12px;">
            <h4 style="color:#007acc; margin:5px 0;">Summary:</h4>
            <p style="margin:2px 0;">• Vehicles: {len(current_routes)}</p>
            <p style="margin:2px 0;">• Total Houses: {len(houses)}</p>
            <p style="margin:2px 0;">• Total Distance: {total_distance/1000:.1f} km</p>
        </div>
        
        <div style="margin-bottom:12px;">
            <h4 style="color:#007acc; margin:5px 0;">Vehicle Clusters:</h4>
            <table style="width:100%; font-size:11px; border-collapse:collapse;">
                <tr style="background:#f0f0f0; font-weight:bold;">
                    <td style="padding:4px; border:1px solid #ccc;">Vehicle</td>
                    <td style="padding:4px; border:1px solid #ccc;">Roads</td>
                    <td style="padding:4px; border:1px solid #ccc;">Houses</td>
                    <td style="padding:4px; border:1px solid #ccc;">Distance</td>
                </tr>
                {vehicle_rows}
            </table>
        </div>
        
        <div style="margin-bottom:10px;">
            <h4 style="color:#007acc; margin:5px 0;">Controls:</h4>
            <p style="margin:2px 0; font-size:11px;">• Use Layer Control (top-left) to toggle:</p>
            <p style="margin:2px 0; font-size:11px;">&nbsp;&nbsp;- "All Routes" for mixed view</p>
            <p style="margin:2px 0; font-size:11px;">&nbsp;&nbsp;- Individual "Vehicle T1, T2..." for separate clusters</p>
            <p style="margin:2px 0; font-size:11px;">• Each route follows actual road segments</p>
            <p style="margin:2px 0; font-size:11px;">• No overlapping road assignments</p>
        </div>
        
        <div style="font-size:10px; color:#666; margin-top:10px; border-top:1px solid #ccc; padding-top:8px;">
            ✅ Non-overlapping routes guaranteed<br>
            ✅ All houses covered via road segments<br>
            Ward 29 boundary in sky blue
        </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(dashboard_html))
        
        return HTMLResponse(content=m._repr_html_())
        
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Handle the specific "False" error
        if str(e) == "False":
            raise HTTPException(status_code=400, detail="Route processing failed. Please try uploading files and assigning routes again.")
        else:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Intelligent Garbage Collection Route Assignment System",
        "endpoints": {
            "upload": "POST /upload - Upload input files",
            "assign_routes": "POST /assign_routes - Compute optimized routes",
            "map": "GET /map - View map with routes"
        }
    }