"""
SmolRouter main entry point - allows running as module with: python -m smolrouter
"""

import os
import sys
import uvicorn
from smolrouter.app import app

def main():
    """Main entry point for SmolRouter"""
    # Get configuration from environment variables
    host = os.getenv("LISTEN_HOST", "127.0.0.1")
    port = int(os.getenv("LISTEN_PORT", "8088"))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    main()