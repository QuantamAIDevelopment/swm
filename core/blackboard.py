"""Blackboard system for shared data and event coordination."""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from models.blackboard_entry import BlackboardEntry, UploadData, RouteResult

logger = logging.getLogger(__name__)

class Blackboard:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._entries: List[BlackboardEntry] = []
        self._lock = threading.Lock()
        
    def store_upload_data(self, upload_id: str, upload_data: UploadData) -> None:
        """Store uploaded data on blackboard."""
        with self._lock:
            self._data[f"upload_{upload_id}"] = upload_data
            entry = BlackboardEntry(
                entry_id=f"upload_{upload_id}",
                entry_type="upload",
                data={"upload_id": upload_id},
                timestamp=datetime.now()
            )
            self._entries.append(entry)
            logger.info(f"Stored upload data for {upload_id}")
    
    def get_upload_data(self, upload_id: str) -> Optional[UploadData]:
        """Retrieve upload data."""
        with self._lock:
            return self._data.get(f"upload_{upload_id}")
    
    def store_routes(self, upload_id: str, routes: List[RouteResult]) -> None:
        """Store computed routes."""
        with self._lock:
            self._data[f"routes_{upload_id}"] = routes
            entry = BlackboardEntry(
                entry_id=f"routes_{upload_id}",
                entry_type="routes",
                data={"upload_id": upload_id, "route_count": len(routes)},
                timestamp=datetime.now(),
                status="completed"
            )
            self._entries.append(entry)
            logger.info(f"Stored {len(routes)} routes for {upload_id}")
    
    def get_routes(self, upload_id: str) -> Optional[List[RouteResult]]:
        """Retrieve computed routes."""
        with self._lock:
            return self._data.get(f"routes_{upload_id}")
    
    def mark_vehicle_unavailable(self, upload_id: str, vehicle_id: str) -> None:
        """Mark a vehicle as unavailable."""
        with self._lock:
            key = f"unavailable_vehicles_{upload_id}"
            if key not in self._data:
                self._data[key] = set()
            self._data[key].add(vehicle_id)
            logger.info(f"Marked vehicle {vehicle_id} as unavailable for {upload_id}")
    
    def get_unavailable_vehicles(self, upload_id: str) -> set:
        """Get set of unavailable vehicles."""
        with self._lock:
            return self._data.get(f"unavailable_vehicles_{upload_id}", set())

# Global blackboard instance
blackboard = Blackboard()