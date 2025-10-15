"""Generate turn-by-turn directions for vehicle routes."""
import math
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DirectionsGenerator:
    """Generate turn-by-turn directions for vehicle routes."""
    
    def __init__(self):
        self.direction_thresholds = {
            'straight': 15,      # <15 degrees
            'slight': 75,        # 15-75 degrees
            'turn': 135,         # 75-135 degrees
            'u_turn': 180        # >135 degrees
        }
        
        self.color_codes = {
            'straight': '🟢',     # Green
            'slight left': '🔵',  # Blue
            'slight right': '🟠', # Orange
            'left turn': '🔵',    # Blue
            'right turn': '🟠',   # Orange
            'u-turn': '🔴',       # Red
            'start': '⚪',        # White
            'arrived': '⚫'       # Black
        }
    
    def calculate_bearing(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate bearing between two points in degrees."""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def calculate_turn_angle(self, bearing1: float, bearing2: float) -> float:
        """Calculate turn angle between two bearings."""
        angle = bearing2 - bearing1
        
        # Normalize to -180 to 180
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
            
        return angle
    
    def get_direction_label(self, turn_angle: float) -> str:
        """Convert turn angle to direction label with refined thresholds."""
        abs_angle = abs(turn_angle)
        
        if abs_angle < self.direction_thresholds['straight']:
            return 'straight'
        elif abs_angle <= self.direction_thresholds['slight']:
            return 'slight left' if turn_angle < 0 else 'slight right'
        elif abs_angle <= self.direction_thresholds['turn']:
            return 'left turn' if turn_angle < 0 else 'right turn'
        else:
            return 'u-turn'
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in meters using Haversine formula."""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        r = 6371000
        
        return c * r
    
    def generate_route_flow_directions(self, route_coords: List[Tuple[float, float]], 
                                     start_point: Tuple[float, float], 
                                     end_point: Tuple[float, float],
                                     vehicle_name: str) -> List[Dict[str, Any]]:
        """Generate directions based on route flow from start to end point."""
        if len(route_coords) < 2:
            return []
        
        # Ensure route starts and ends at specified points
        flow_coords = [start_point] + route_coords + [end_point]
        
        # Remove duplicate consecutive points
        cleaned_coords = [flow_coords[0]]
        for coord in flow_coords[1:]:
            if self.calculate_distance(cleaned_coords[-1], coord) > 5:  # 5m threshold
                cleaned_coords.append(coord)
        
        directions = []
        total_distance = 0
        
        # Start instruction
        directions.append({
            'step': 1,
            'instruction': f"Start route at {vehicle_name} depot",
            'direction': 'start',
            'color': self.color_codes['start'],
            'coordinates': cleaned_coords[0],
            'distance_meters': 0,
            'cumulative_distance': 0,
            'bearing': None
        })
        
        # Calculate bearings for the flow
        bearings = []
        for i in range(len(cleaned_coords) - 1):
            bearing = self.calculate_bearing(cleaned_coords[i], cleaned_coords[i + 1])
            bearings.append(bearing)
        
        # Generate flow directions
        for i in range(1, len(cleaned_coords)):
            current_coords = cleaned_coords[i]
            prev_coords = cleaned_coords[i - 1]
            
            segment_distance = self.calculate_distance(prev_coords, current_coords)
            total_distance += segment_distance
            
            if i == 1:
                # First segment
                compass_dir = self._bearing_to_compass(bearings[0])
                direction = 'straight'
                instruction = f"Head {compass_dir} on route"
            elif i == len(cleaned_coords) - 1:
                # Last segment to endpoint
                direction = 'straight'
                instruction = f"Continue to endpoint"
            else:
                # Calculate turn
                prev_bearing = bearings[i - 2]
                current_bearing = bearings[i - 1]
                turn_angle = self.calculate_turn_angle(prev_bearing, current_bearing)
                direction = self.get_direction_label(turn_angle)
                
                if direction == 'straight':
                    instruction = "Continue straight"
                elif 'slight' in direction:
                    instruction = f"Make a {direction}"
                elif 'turn' in direction:
                    instruction = f"Turn {direction.split()[0]}"
                else:
                    instruction = "Make a U-turn"
            
            directions.append({
                'step': i + 1,
                'instruction': instruction,
                'direction': direction,
                'color': self.color_codes.get(direction, '⚪'),
                'coordinates': current_coords,
                'distance_meters': round(segment_distance, 1),
                'cumulative_distance': round(total_distance, 1),
                'bearing': bearings[i - 1] if i <= len(bearings) else None
            })
        
        # Arrival instruction
        directions.append({
            'step': len(directions) + 1,
            'instruction': f"Arrive at {vehicle_name} endpoint - route complete",
            'direction': 'arrived',
            'color': self.color_codes['arrived'],
            'coordinates': cleaned_coords[-1],
            'distance_meters': 0,
            'cumulative_distance': round(total_distance, 1),
            'bearing': None
        })
        
        return directions
    
    def _bearing_to_compass(self, bearing: float) -> str:
        """Convert bearing to compass direction."""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        
        # Normalize bearing to 0-360
        bearing = bearing % 360
        
        # Calculate index (8 directions, 45 degrees each)
        index = round(bearing / 45) % 8
        
        return directions[index]
    
    def generate_route_summary(self, directions: List[Dict[str, Any]], 
                             vehicle_name: str) -> Dict[str, Any]:
        """Generate a summary of the route directions."""
        if not directions:
            return {}
        
        total_distance = directions[-1]['cumulative_distance']
        total_steps = len([d for d in directions if d['direction'] not in ['start', 'arrived']])
        
        # Count turn types
        turn_counts = {}
        for direction in directions:
            dir_type = direction['direction']
            if dir_type not in ['start', 'arrived']:
                turn_counts[dir_type] = turn_counts.get(dir_type, 0) + 1
        
        return {
            'vehicle_name': vehicle_name,
            'total_distance_meters': total_distance,
            'total_distance_km': round(total_distance / 1000, 2),
            'total_steps': total_steps,
            'turn_summary': turn_counts,
            'start_coordinates': directions[0]['coordinates'],
            'end_coordinates': directions[-1]['coordinates']
        }
    
    def format_directions_text(self, directions: List[Dict[str, Any]]) -> str:
        """Format directions as readable text with color codes."""
        if not directions:
            return "No directions available."
        
        text_lines = []
        for direction in directions:
            step = direction['step']
            instruction = direction['instruction']
            distance = direction['distance_meters']
            color = direction.get('color', '⚪')
            
            if direction['direction'] in ['start', 'arrived']:
                text_lines.append(f"{step}. {color} {instruction}")
            else:
                text_lines.append(f"{step}. {color} {instruction} ({distance}m)")
        
        return "\n".join(text_lines)