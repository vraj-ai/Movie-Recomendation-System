"""``python -m recsys.web`` — run the taste UI locally.

Data mode is explicit: pass ``--fixture`` or ``--real`` (WEB-FAIL-1).
Binds 127.0.0.1 by default (WEB-SEC-1).
"""

from __future__ import annotations

import argparse
import sys

from recsys.config import WEB_HOST, WEB_PORT  # P3-6: single source of truth


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m recsys.web")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true",
                      help="serve the bundled fixture dataset")
    mode.add_argument("--real", action="store_true",
                      help="serve data/ml-latest-small (must exist)")
    p.add_argument("--host", default=WEB_HOST,
                   help=f"bind address (default {WEB_HOST}, WEB-SEC-1)")
    p.add_argument("--port", type=int, default=WEB_PORT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import os

    import uvicorn
    from dotenv import load_dotenv

    from recsys.web.app import create_app

    load_dotenv()
    try:
        app = create_app(
            mode="fixture" if args.fixture else "real",
            tmdb_api_key=os.environ.get("TMDB_API_KEY"),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
