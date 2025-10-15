from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import folium
import json
import numpy as np
from io import BytesIO
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Waste Management System",
    description="Simplified Garbage Collection Route Assignment System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ward_gdf = None
roads_gdf = None
buildings_gdf = None

@app.post("/upload")
async def upload_files(
    ward_geojson: UploadFile = File(...),
    roads_geojson: UploadFile = File(...),
    houses_geojson: UploadFile = File(...),
    vehicles_csv: UploadFile = File(...)
):
    global ward_gdf, roads_gdf, buildings_gdf
    
    try:
        import geopandas as gpd
        
        ward_content = await ward_geojson.read()
        roads_content = await roads_geojson.read()
        houses_content = await houses_geojson.read()
        
        ward_gdf = gpd.read_file(BytesIO(ward_content))
        roads_gdf = gpd.read_file(BytesIO(roads_content))
        buildings_gdf = gpd.read_file(BytesIO(houses_content))
        
        logger.info(f"Uploaded: Ward={len(ward_gdf)}, Roads={len(roads_gdf)}, Buildings={len(buildings_gdf)}")
        
        return {
            "status": "success",
            "message": "Files uploaded successfully",
            "ward_features": len(ward_gdf),
            "roads_features": len(roads_gdf),
            "buildings_features": len(buildings_gdf)
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing files: {str(e)}")

@app.get("/test-map/", response_class=HTMLResponse)
async def test_map():
    """Simple test map without data dependencies"""
    try:
        # Create a basic map centered on Bangalore
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
        
        # Add some test markers
        test_locations = [
            [12.9716, 77.5946, "Location 1"],
            [12.9800, 77.6000, "Location 2"],
            [12.9650, 77.5900, "Location 3"]
        ]
        
        colors = ['red', 'blue', 'green']
        for i, (lat, lon, name) in enumerate(test_locations):
            folium.Marker(
                [lat, lon],
                popup=name,
                icon=folium.Icon(color=colors[i % len(colors)])
            ).add_to(m)
        
        return HTMLResponse(content=m._repr_html_())
    except Exception as e:
        logger.error(f"Test map error: {str(e)}")
        return HTMLResponse(content=f"<h1>Error creating test map: {str(e)}</h1>")

@app.get("/generate-map/", response_class=HTMLResponse)
async def generate_map():
    """Generate map with uploaded data"""
    try:
        # Check if data exists
        if ward_gdf is None or buildings_gdf is None:
            return HTMLResponse(content="""
            <h1>No Data Uploaded</h1>
            <p>Please upload data files first using the <a href="/docs">/upload</a> endpoint.</p>
            <p><a href="/test-map/">View Test Map</a></p>
            """)
        
        logger.info("Starting map generation...")
        
        # Convert to WGS84 if needed
        ward_wgs = ward_gdf.to_crs('EPSG:4326') if ward_gdf.crs != 'EPSG:4326' else ward_gdf
        buildings_wgs = buildings_gdf.to_crs('EPSG:4326') if buildings_gdf.crs != 'EPSG:4326' else buildings_gdf
        
        # Get center from ward bounds
        bounds = ward_wgs.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        logger.info(f"Map center: {center_lat}, {center_lon}")
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Add ward boundaries
        folium.GeoJson(
            json.loads(ward_wgs.to_json()),
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)
        
        # Simple clustering - divide buildings into 3 groups
        building_centroids = [(pt.x, pt.y) for pt in buildings_wgs.geometry.centroid]
        
        if len(building_centroids) > 0:
            from sklearn.cluster import KMeans
            
            n_clusters = min(3, len(building_centroids))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(building_centroids)
            
            colors = ['red', 'blue', 'green']
            
            # Add buildings with cluster colors
            for i, (building, cluster) in enumerate(zip(buildings_wgs.itertuples(), clusters)):
                color = colors[cluster % len(colors)]
                
                # Add building marker
                centroid = building.geometry.centroid
                folium.CircleMarker(
                    [centroid.y, centroid.x],
                    radius=5,
                    popup=f"Building {i+1} - Cluster {cluster+1}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        logger.info("Map generation completed successfully")
        return HTMLResponse(content=m._repr_html_())
        
    except Exception as e:
        logger.error(f"Map generation error: {str(e)}")
        logger.error(traceback.format_exc())
        return HTMLResponse(content=f"""
        <h1>Map Generation Error</h1>
        <p>Error: {str(e)}</p>
        <p><a href="/test-map/">Try Test Map</a></p>
        <p><a href="/docs">API Documentation</a></p>
        """)

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <head><title>Smart Waste Management</title></head>
        <body>
            <h2>🚛 Smart Waste Management System</h2>
            <div style="margin: 20px 0;">
                <h3>Available Endpoints:</h3>
                <ul>
                    <li><a href="/test-map/">🗺️ Test Map (No data required)</a></li>
                    <li><a href="/docs">📚 Upload Data & Generate Map</a></li>
                    <li><a href="/generate-map/">🗺️ Generate Map (After upload)</a></li>
                </ul>
            </div>
        </body>
    </html>
    """)