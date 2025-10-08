"""Main entry point for the garbage collection route assignment system."""
import logging
import os
import socket
import uvicorn
from api.routes import app
from configurations.config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def is_port_available(host, port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def main():
    """Start the FastAPI application."""
    logger.info("Starting Intelligent Garbage Collection Route Assignment System")
    
    # Try different ports if the configured one is busy
    ports_to_try = [Config.API_PORT, 7001, 7002, 7003, 7004, 7005, 6000, 6001, 6002]
    
    available_port = None
    for port in ports_to_try:
        if is_port_available(Config.API_HOST, port):
            available_port = port
            break
        else:
            logger.warning(f"Port {port} is busy")
    
    if available_port is None:
        logger.error("Could not find an available port")
        raise Exception("No available ports found")
    
    logger.info(f"Starting server on {Config.API_HOST}:{available_port}")
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=available_port,
        log_level=Config.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()