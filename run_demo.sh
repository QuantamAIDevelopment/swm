#!/bin/bash

# Demo script for Intelligent Garbage Collection Route Assignment System

echo "=== Intelligent Garbage Collection Route Assignment System Demo ==="
echo

# Create sample data directory
mkdir -p sample_data

# Generate sample ward boundaries GeoJSON
cat > sample_data/ward_boundaries.geojson << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "ward_id": 1,
        "ward_name": "Ward 1"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1000, 0], [1000, 1000], [0, 1000], [0, 0]]]
      }
    }
  ]
}
EOF

# Generate sample road network GeoJSON
cat > sample_data/road_network.geojson << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"road_id": "R001"},
      "geometry": {"type": "LineString", "coordinates": [[0, 250], [1000, 250]]}
    },
    {
      "type": "Feature", 
      "properties": {"road_id": "R002"},
      "geometry": {"type": "LineString", "coordinates": [[0, 500], [1000, 500]]}
    },
    {
      "type": "Feature",
      "properties": {"road_id": "R003"}, 
      "geometry": {"type": "LineString", "coordinates": [[0, 750], [1000, 750]]}
    },
    {
      "type": "Feature",
      "properties": {"road_id": "R004"},
      "geometry": {"type": "LineString", "coordinates": [[250, 0], [250, 1000]]}
    },
    {
      "type": "Feature",
      "properties": {"road_id": "R005"},
      "geometry": {"type": "LineString", "coordinates": [[500, 0], [500, 1000]]}
    },
    {
      "type": "Feature",
      "properties": {"road_id": "R006"},
      "geometry": {"type": "LineString", "coordinates": [[750, 0], [750, 1000]]}
    }
  ]
}
EOF

# Generate sample houses GeoJSON (60 houses in grid pattern)
cat > sample_data/houses.geojson << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"house_id": "H001"}, "geometry": {"type": "Point", "coordinates": [100, 200]}},
    {"type": "Feature", "properties": {"house_id": "H002"}, "geometry": {"type": "Point", "coordinates": [200, 200]}},
    {"type": "Feature", "properties": {"house_id": "H003"}, "geometry": {"type": "Point", "coordinates": [300, 200]}},
    {"type": "Feature", "properties": {"house_id": "H004"}, "geometry": {"type": "Point", "coordinates": [400, 200]}},
    {"type": "Feature", "properties": {"house_id": "H005"}, "geometry": {"type": "Point", "coordinates": [500, 200]}},
    {"type": "Feature", "properties": {"house_id": "H006"}, "geometry": {"type": "Point", "coordinates": [600, 200]}},
    {"type": "Feature", "properties": {"house_id": "H007"}, "geometry": {"type": "Point", "coordinates": [700, 200]}},
    {"type": "Feature", "properties": {"house_id": "H008"}, "geometry": {"type": "Point", "coordinates": [800, 200]}},
    {"type": "Feature", "properties": {"house_id": "H009"}, "geometry": {"type": "Point", "coordinates": [900, 200]}},
    {"type": "Feature", "properties": {"house_id": "H010"}, "geometry": {"type": "Point", "coordinates": [100, 300]}},
    {"type": "Feature", "properties": {"house_id": "H011"}, "geometry": {"type": "Point", "coordinates": [200, 300]}},
    {"type": "Feature", "properties": {"house_id": "H012"}, "geometry": {"type": "Point", "coordinates": [300, 300]}},
    {"type": "Feature", "properties": {"house_id": "H013"}, "geometry": {"type": "Point", "coordinates": [400, 300]}},
    {"type": "Feature", "properties": {"house_id": "H014"}, "geometry": {"type": "Point", "coordinates": [500, 300]}},
    {"type": "Feature", "properties": {"house_id": "H015"}, "geometry": {"type": "Point", "coordinates": [600, 300]}},
    {"type": "Feature", "properties": {"house_id": "H016"}, "geometry": {"type": "Point", "coordinates": [700, 300]}},
    {"type": "Feature", "properties": {"house_id": "H017"}, "geometry": {"type": "Point", "coordinates": [800, 300]}},
    {"type": "Feature", "properties": {"house_id": "H018"}, "geometry": {"type": "Point", "coordinates": [900, 300]}},
    {"type": "Feature", "properties": {"house_id": "H019"}, "geometry": {"type": "Point", "coordinates": [100, 400]}},
    {"type": "Feature", "properties": {"house_id": "H020"}, "geometry": {"type": "Point", "coordinates": [200, 400]}},
    {"type": "Feature", "properties": {"house_id": "H021"}, "geometry": {"type": "Point", "coordinates": [300, 400]}},
    {"type": "Feature", "properties": {"house_id": "H022"}, "geometry": {"type": "Point", "coordinates": [400, 400]}},
    {"type": "Feature", "properties": {"house_id": "H023"}, "geometry": {"type": "Point", "coordinates": [500, 400]}},
    {"type": "Feature", "properties": {"house_id": "H024"}, "geometry": {"type": "Point", "coordinates": [600, 400]}},
    {"type": "Feature", "properties": {"house_id": "H025"}, "geometry": {"type": "Point", "coordinates": [700, 400]}},
    {"type": "Feature", "properties": {"house_id": "H026"}, "geometry": {"type": "Point", "coordinates": [800, 400]}},
    {"type": "Feature", "properties": {"house_id": "H027"}, "geometry": {"type": "Point", "coordinates": [900, 400]}},
    {"type": "Feature", "properties": {"house_id": "H028"}, "geometry": {"type": "Point", "coordinates": [100, 600]}},
    {"type": "Feature", "properties": {"house_id": "H029"}, "geometry": {"type": "Point", "coordinates": [200, 600]}},
    {"type": "Feature", "properties": {"house_id": "H030"}, "geometry": {"type": "Point", "coordinates": [300, 600]}},
    {"type": "Feature", "properties": {"house_id": "H031"}, "geometry": {"type": "Point", "coordinates": [400, 600]}},
    {"type": "Feature", "properties": {"house_id": "H032"}, "geometry": {"type": "Point", "coordinates": [500, 600]}},
    {"type": "Feature", "properties": {"house_id": "H033"}, "geometry": {"type": "Point", "coordinates": [600, 600]}},
    {"type": "Feature", "properties": {"house_id": "H034"}, "geometry": {"type": "Point", "coordinates": [700, 600]}},
    {"type": "Feature", "properties": {"house_id": "H035"}, "geometry": {"type": "Point", "coordinates": [800, 600]}},
    {"type": "Feature", "properties": {"house_id": "H036"}, "geometry": {"type": "Point", "coordinates": [900, 600]}},
    {"type": "Feature", "properties": {"house_id": "H037"}, "geometry": {"type": "Point", "coordinates": [100, 700]}},
    {"type": "Feature", "properties": {"house_id": "H038"}, "geometry": {"type": "Point", "coordinates": [200, 700]}},
    {"type": "Feature", "properties": {"house_id": "H039"}, "geometry": {"type": "Point", "coordinates": [300, 700]}},
    {"type": "Feature", "properties": {"house_id": "H040"}, "geometry": {"type": "Point", "coordinates": [400, 700]}},
    {"type": "Feature", "properties": {"house_id": "H041"}, "geometry": {"type": "Point", "coordinates": [500, 700]}},
    {"type": "Feature", "properties": {"house_id": "H042"}, "geometry": {"type": "Point", "coordinates": [600, 700]}},
    {"type": "Feature", "properties": {"house_id": "H043"}, "geometry": {"type": "Point", "coordinates": [700, 700]}},
    {"type": "Feature", "properties": {"house_id": "H044"}, "geometry": {"type": "Point", "coordinates": [800, 700]}},
    {"type": "Feature", "properties": {"house_id": "H045"}, "geometry": {"type": "Point", "coordinates": [900, 700]}},
    {"type": "Feature", "properties": {"house_id": "H046"}, "geometry": {"type": "Point", "coordinates": [100, 800]}},
    {"type": "Feature", "properties": {"house_id": "H047"}, "geometry": {"type": "Point", "coordinates": [200, 800]}},
    {"type": "Feature", "properties": {"house_id": "H048"}, "geometry": {"type": "Point", "coordinates": [300, 800]}},
    {"type": "Feature", "properties": {"house_id": "H049"}, "geometry": {"type": "Point", "coordinates": [400, 800]}},
    {"type": "Feature", "properties": {"house_id": "H050"}, "geometry": {"type": "Point", "coordinates": [500, 800]}},
    {"type": "Feature", "properties": {"house_id": "H051"}, "geometry": {"type": "Point", "coordinates": [600, 800]}},
    {"type": "Feature", "properties": {"house_id": "H052"}, "geometry": {"type": "Point", "coordinates": [700, 800]}},
    {"type": "Feature", "properties": {"house_id": "H053"}, "geometry": {"type": "Point", "coordinates": [800, 800]}},
    {"type": "Feature", "properties": {"house_id": "H054"}, "geometry": {"type": "Point", "coordinates": [900, 800]}},
    {"type": "Feature", "properties": {"house_id": "H055"}, "geometry": {"type": "Point", "coordinates": [150, 150]}},
    {"type": "Feature", "properties": {"house_id": "H056"}, "geometry": {"type": "Point", "coordinates": [350, 150]}},
    {"type": "Feature", "properties": {"house_id": "H057"}, "geometry": {"type": "Point", "coordinates": [550, 150]}},
    {"type": "Feature", "properties": {"house_id": "H058"}, "geometry": {"type": "Point", "coordinates": [750, 150]}},
    {"type": "Feature", "properties": {"house_id": "H059"}, "geometry": {"type": "Point", "coordinates": [150, 850]}},
    {"type": "Feature", "properties": {"house_id": "H060"}, "geometry": {"type": "Point", "coordinates": [850, 850]}}
  ]
}
EOF

# Generate sample vehicles CSV
cat > sample_data/vehicles.csv << 'EOF'
vehicle_id,vehicle_type,ward_no,driver_info,status,start_location
V001,truck,1,John Doe,active,250,250
V002,truck,1,Jane Smith,active,500,500
V003,truck,1,Bob Johnson,active,750,750
EOF

echo "Sample data generated in sample_data/ directory"
echo

# Start the API server in background
echo "Starting API server..."
python main.py &
API_PID=$!

# Wait for server to start
sleep 5

echo "API server started (PID: $API_PID)"
echo "Swagger UI available at: http://localhost:8000/docs"
echo

# Upload sample data
echo "Uploading sample data..."
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/upload" \
  -F "ward_geojson=@sample_data/ward_boundaries.geojson" \
  -F "roads_geojson=@sample_data/road_network.geojson" \
  -F "houses_geojson=@sample_data/houses.geojson" \
  -F "vehicles_csv=@sample_data/vehicles.csv")

echo "Upload response: $UPLOAD_RESPONSE"

# Extract upload_id (assuming JSON response)
UPLOAD_ID=$(echo $UPLOAD_RESPONSE | grep -o '"upload_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$UPLOAD_ID" ]; then
    echo "Error: Could not extract upload_id from response"
    kill $API_PID
    exit 1
fi

echo "Upload ID: $UPLOAD_ID"
echo

# Compute routes
echo "Computing routes..."
COMPUTE_RESPONSE=$(curl -s -X POST "http://localhost:8000/compute_routes" \
  -H "Content-Type: application/json" \
  -d "{\"upload_id\": \"$UPLOAD_ID\"}")

echo "Compute response: $COMPUTE_RESPONSE"
echo

# Wait for processing
echo "Waiting for route computation to complete..."
sleep 10

# Get routes
echo "Fetching computed routes..."
ROUTES_RESPONSE=$(curl -s "http://localhost:8000/routes/$UPLOAD_ID")

echo "Routes computed successfully!"
echo

# Count features in response
ROUTE_COUNT=$(echo $ROUTES_RESPONSE | grep -o '"type":"Feature"' | wc -l)
echo "Generated $ROUTE_COUNT routes"

# Show route summary
echo
echo "=== ROUTE SUMMARY ==="
echo $ROUTES_RESPONSE | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for i, feature in enumerate(data.get('features', [])):
        props = feature.get('properties', {})
        print(f'Route {i+1}:')
        print(f'  Vehicle: {props.get(\"vehicle_id\", \"N/A\")}')
        print(f'  Houses: {len(props.get(\"ordered_house_ids\", []))}')
        print(f'  Distance: {props.get(\"total_distance_meters\", 0):.1f}m')
        print()
except:
    print('Could not parse route data')
"

echo
echo "=== VISUALIZATION ==="
echo "Folium map preview available at:"
echo "http://localhost:8000/preview/$UPLOAD_ID"
echo
echo "GeoJSON routes available at:"
echo "http://localhost:8000/routes/$UPLOAD_ID"
echo

echo "Demo completed successfully!"
echo "Press Ctrl+C to stop the API server"

# Keep server running
wait $API_PID
EOF