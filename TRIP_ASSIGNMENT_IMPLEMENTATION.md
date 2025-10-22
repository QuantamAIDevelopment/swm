# Vehicle Capacity and Trip Assignment Implementation

## Overview
This implementation adds dynamic trip assignment logic to the SWM garbage collection system based on vehicle capacity constraints and ensures no overlap between trips.

## Key Features Implemented

### 1. Vehicle Capacity Configuration
- **Houses per vehicle per trip**: 500 houses maximum
- **Maximum trips per day**: 2 trips per vehicle
- Configuration stored in `configurations/config.py`

### 2. Dynamic Trip Assignment Logic
- **Single Trip**: If total houses ≤ (active vehicles × 500), assign 1 trip
- **Multiple Trips**: If total houses > single trip capacity, assign up to 2 trips
- **Equal Distribution**: Houses evenly distributed among vehicles within each trip

### 3. No Overlap Rule
- Each house assigned to exactly one trip
- Validation ensures no house appears in multiple trips
- Sequential trip processing prevents route conflicts

## Files Modified/Created

### New Files
1. **`clustering/trip_assignment.py`** - Core trip assignment logic
2. **`test_trip_assignment.py`** - Test suite for validation

### Modified Files
1. **`configurations/config.py`** - Added capacity constants
2. **`clustering/assign_buildings.py`** - Integrated trip assignment
3. **`routing/compute_routes.py`** - Updated for multi-trip routing
4. **`main.py`** - Added trip statistics logging

## Implementation Details

### Trip Assignment Algorithm
```python
# Determine trips needed
total_capacity_single_trip = num_vehicles * 500
if total_houses <= total_capacity_single_trip:
    num_trips = 1
else:
    num_trips = min(2, ceil(total_houses / total_capacity_single_trip))

# Distribute houses across trips (no overlap)
houses_per_trip = total_houses // num_trips
for trip in range(num_trips):
    # Assign distinct house subset to each trip
    # Distribute houses among vehicles within trip
```

### Route Computation
- Routes computed separately for each trip
- Trip 1 completes before Trip 2 begins
- No geographical overlap between trip routes
- Vehicles can make up to 2 trips per day

## Usage Examples

### Scenario 1: Single Trip (1200 houses, 3 vehicles)
- Capacity: 3 × 500 = 1500 houses
- Result: 1 trip, 400 houses per vehicle

### Scenario 2: Multiple Trips (1200 houses, 2 vehicles)  
- Capacity: 2 × 500 = 1000 houses per trip
- Result: 2 trips, 600 houses per trip (300 per vehicle)

## Testing
Run the test suite to validate implementation:
```bash
python test_trip_assignment.py
```

Tests cover:
- Single vs multiple trip scenarios
- Equal distribution among vehicles
- No overlap validation
- Configuration verification

## Integration
The trip assignment is automatically integrated into the main pipeline:
```bash
python main.py --roads roads.geojson --buildings buildings.geojson
```

Trip statistics are logged and included in results output.

## Benefits
1. **Capacity Management**: Ensures vehicles don't exceed 500 houses per trip
2. **Efficiency**: Optimal distribution of workload across vehicles
3. **No Conflicts**: Prevents route overlap between trips
4. **Scalability**: Handles varying numbers of houses and vehicles
5. **Validation**: Built-in checks ensure data integrity