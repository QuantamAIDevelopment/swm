"""PostGIS storage for route geometries and data."""
import logging
from typing import List, Optional
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from models.blackboard_entry import RouteResult
from configurations.config import Config

logger = logging.getLogger(__name__)

class PostGISStore:
    def __init__(self):
        try:
            self.engine = create_engine(Config.DATABASE_URL)
            self._create_tables()
        except Exception as e:
            logger.warning(f"PostGIS connection failed: {e}. Running without database.")
            self.engine = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            with self.engine.connect() as conn:
                # Create routes table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS routes (
                        id SERIAL PRIMARY KEY,
                        upload_id VARCHAR(255),
                        vehicle_id VARCHAR(255),
                        route_id VARCHAR(255),
                        ordered_house_ids TEXT[],
                        road_segment_ids TEXT[],
                        start_node VARCHAR(255),
                        end_node VARCHAR(255),
                        total_distance_meters FLOAT,
                        status VARCHAR(50),
                        geometry GEOMETRY(LINESTRING, 3857),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create spatial index
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_routes_geometry 
                    ON routes USING GIST (geometry)
                """))
                
                conn.commit()
                logger.info("PostGIS tables created successfully")
        except Exception as e:
            logger.error(f"Error creating PostGIS tables: {e}")
    
    def store_routes(self, upload_id: str, routes: List[RouteResult]) -> bool:
        """Store routes in PostGIS database."""
        if self.engine is None:
            logger.warning("No database connection, skipping route storage")
            return True
        try:
            # Convert routes to GeoDataFrame
            route_data = []
            for route in routes:
                route_data.append({
                    'upload_id': upload_id,
                    'vehicle_id': route.vehicle_id,
                    'route_id': route.route_id,
                    'ordered_house_ids': route.ordered_house_ids,
                    'road_segment_ids': route.road_segment_ids,
                    'start_node': route.start_node,
                    'end_node': route.end_node,
                    'total_distance_meters': route.total_distance_meters,
                    'status': route.status,
                    'geometry': route.geometry
                })
            
            if not route_data:
                return True
            
            routes_gdf = gpd.GeoDataFrame(route_data, crs=Config.TARGET_CRS)
            
            # Store in PostGIS
            routes_gdf.to_postgis(
                'routes', 
                self.engine, 
                if_exists='append', 
                index=False,
                dtype={'geometry': Geometry('LINESTRING', srid=3857)}
            )
            
            logger.info(f"Stored {len(routes)} routes for upload {upload_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing routes: {e}")
            return False
    
    def get_routes(self, upload_id: str) -> Optional[gpd.GeoDataFrame]:
        """Retrieve routes from PostGIS."""
        try:
            query = """
                SELECT * FROM routes 
                WHERE upload_id = %s 
                ORDER BY vehicle_id
            """
            
            routes_gdf = gpd.read_postgis(
                query, 
                self.engine, 
                params=[upload_id],
                geom_col='geometry'
            )
            
            return routes_gdf if not routes_gdf.empty else None
            
        except Exception as e:
            logger.error(f"Error retrieving routes: {e}")
            return None
    
    def delete_routes(self, upload_id: str) -> bool:
        """Delete routes for a specific upload."""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM routes WHERE upload_id = :upload_id"),
                    {"upload_id": upload_id}
                )
                conn.commit()
            
            logger.info(f"Deleted routes for upload {upload_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting routes: {e}")
            return False