"""Data models for blackboard entries."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import geopandas as gpd
import pandas as pd

@dataclass
class UploadData:
    upload_id: str
    ward_boundaries: gpd.GeoDataFrame
    road_network: gpd.GeoDataFrame
    houses: gpd.GeoDataFrame
    vehicles: pd.DataFrame
    timestamp: datetime

@dataclass
class RouteResult:
    vehicle_id: str
    route_id: str
    ordered_house_ids: List[str]
    road_segment_ids: List[str]
    start_node: str
    end_node: str
    total_distance_meters: float
    status: str
    geometry: Any  # LineString geometry

@dataclass
class BlackboardEntry:
    entry_id: str
    entry_type: str
    data: Dict[str, Any]
    timestamp: datetime
    status: str = "pending"