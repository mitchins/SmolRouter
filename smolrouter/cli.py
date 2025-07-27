import uvicorn
from smolrouter.app import app, LISTEN_HOST, LISTEN_PORT

def main():
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
