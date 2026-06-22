import argparse
import os

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SmolRouter")
    parser.add_argument("-C", "--config", dest="routes_config", help="Path to routes.yaml or routes.json")
    parser.add_argument("--host", default=os.getenv("LISTEN_HOST", "127.0.0.1"), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("LISTEN_PORT", "1234")), help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.routes_config:
        os.environ["ROUTES_CONFIG"] = args.routes_config

    from smolrouter.app import app, configure_logging

    configure_logging()

    os.environ["LISTEN_HOST"] = args.host
    os.environ["LISTEN_PORT"] = str(args.port)
    if args.reload:
        os.environ["RELOAD"] = "true"

    app_target = "smolrouter.app:app" if args.reload else None
    if app_target is not None:
        uvicorn.run(app_target, host=args.host, port=args.port, reload=True)
        return

    uvicorn.run(app, host=args.host, port=args.port)
