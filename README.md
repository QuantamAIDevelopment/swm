# Intelligent Garbage Collection Route Assignment System

A production-ready Python system for optimizing garbage collection routes using agent-driven architecture, clustering, and VRP solving.

## Features

- **Multi-file Input Processing**: Supports Ward Boundaries, Road Network, Houses (GeoJSON) and Vehicles (CSV)
- **Intelligent Route Optimization**: Uses KMeans clustering and OR-Tools VRP solver
- **Non-overlapping Routes**: Ensures each road segment is assigned to exactly one vehicle
- **Dynamic Reassignment**: Handles vehicle unavailability with automatic route redistribution
- **PostGIS Storage**: Persists route geometries and data
- **Interactive Visualization**: Folium-based web preview and GeoJSON API
- **RESTful API**: OpenAPI/Swagger interface for all operations

## Architecture

```
├── agents/                 # Route assignment logic
├── api/                   # FastAPI endpoints
├── configurations/        # System configuration
├── core/                  # Blackboard coordination system
├── models/                # Data models
├── storage/               # PostGIS integration
├── tools/                 # Road snapping and VRP solving
└── tests/                 # Unit and integration tests
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL with PostGIS extension
- Git

### Installation

1. Clone and setup:
```bash
git clone <repository>
cd waste
pip install -r requirements.txt
```

2. Configure database:
```bash
# Set environment variable
export DATABASE_URL="postgresql://user:password@localhost:5432/waste_db"
```

3. Run the system:
```bash
python main.py
```

The API will be available at `http://localhost:8000` with Swagger UI at `http://localhost:8000/docs`.

## API Usage

### 1. Upload Files
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "ward_geojson=@ward_boundaries.geojson" \
  -F "roads_geojson=@road_network.geojson" \
  -F "houses_geojson=@houses.geojson" \
  -F "vehicles_csv=@vehicles.csv"
```

### 2. Compute Routes
```bash
curl -X POST "http://localhost:8000/compute_routes" \
  -H "Content-Type: application/json" \
  -d '{"upload_id": "your-upload-id"}'
```

### 3. Get Routes
```bash
curl "http://localhost:8000/routes/{upload_id}"
```

### 4. Mark Vehicle Unavailable
```bash
curl -X POST "http://localhost:8000/vehicle/{vehicle_id}/mark_unavailable" \
  -H "Content-Type: application/json" \
  -d '{"upload_id": "your-upload-id", "timestamp": "2024-01-01T10:00:00"}'
```

### 5. View Map Preview
Navigate to: `http://localhost:8000/preview/{upload_id}`

## Input File Formats

### Ward Boundaries (GeoJSON)
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"ward_id": 1, "ward_name": "Ward 1"},
    "geometry": {"type": "Polygon", "coordinates": [...]}
  }]
}
```

### Road Network (GeoJSON)
```json
{
  "type": "FeatureCollection", 
  "features": [{
    "type": "Feature",
    "properties": {"road_id": "R001", "road_name": "Main St"},
    "geometry": {"type": "LineString", "coordinates": [...]}
  }]
}
```

### Houses (GeoJSON)
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", 
    "properties": {"house_id": "H001", "ward_no": 1},
    "geometry": {"type": "Point", "coordinates": [...]}
  }]
}
```

### Vehicles (CSV)
```csv
vehicle_id,vehicle_type,ward_no,driver_info,status,start_location
V001,truck,1,John Doe,active,"12.34,56.78"
V002,truck,1,Jane Smith,unavailable,
```

## Algorithm Details

### 1. Preprocessing
- Reprojects all spatial data to EPSG:3857 (Web Mercator)
- Snaps houses to nearest road segments
- Builds routable graph from road network

### 2. Clustering
- Uses KMeans with k = number of active vehicles
- Assigns clusters to vehicles based on proximity
- Ensures road segments belong to single clusters

### 3. Route Optimization
- Builds VRP distance matrix using road network shortest paths
- Solves using OR-Tools with distance minimization objective
- Generates ordered route polylines following road geometry

### 4. Conflict Resolution
- Detects overlapping road segments between routes
- Reassigns disputed segments to minimize cost increase
- Rebuilds affected routes

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run integration test:
```bash
pytest tests/test_snapper_vrp.py::TestIntegration::test_end_to_end_small_dataset -v
```

## Configuration

Edit `configurations/config.py`:

```python
class Config:
    DATABASE_URL = "postgresql://user:pass@localhost:5432/waste_db"
    TARGET_CRS = "EPSG:3857"
    RANDOM_SEED = 42
    VRP_TIME_LIMIT_SECONDS = 30
    API_HOST = "0.0.0.0"
    API_PORT = 8000
```

## Performance

- Optimized for ward sizes up to several thousand houses
- Uses clustering to limit VRP problem size
- Deterministic results with configurable random seed
- Reasonable performance through spatial indexing

## Logging

The system provides comprehensive logging:
- Clustering assignments and decisions
- Road segment assignment rationale  
- OR-Tools objective values
- Conflict resolution steps
- Performance metrics

## Docker Support

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## License

MIT License - see LICENSE file for details.