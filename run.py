"""
Convenience launcher.

Usage:
    python run.py                       # 127.0.0.1:8000
    python run.py --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the AGOP Decision Engine")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="bind port (default 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="enable auto-reload (development)")
    args = parser.parse_args()

    uvicorn.run(
        "agop_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()