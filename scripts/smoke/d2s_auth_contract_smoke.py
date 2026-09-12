"""Smoke-test the public Desktop2Stereo authorization contract without an account."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop2stereo.auth.client import AuthClient, AuthError


def run_smoke(base_url: str) -> dict[str, object]:
    client = AuthClient(base_url.rstrip("/"))
    keys = client.public_keys()
    authorization = client.authorize_device()
    try:
        try:
            client.device_token(authorization.device_code)
        except AuthError as error:
            if error.code != "authorization_pending":
                raise
        else:
            raise RuntimeError("device authorization unexpectedly completed without user approval")
    finally:
        client.cancel_device(authorization.device_code)
    return {
        "public_keys": len(keys),
        "key_id": keys[0]["kid"],
        "device_authorization": "pending",
        "cancel": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("D2S_API_BASE_URL", "https://100393.com/api/v1"),
        help="D2S versioned API origin",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print machine-readable output")
    args = parser.parse_args()
    try:
        result = run_smoke(args.base_url)
    except (AuthError, RuntimeError) as error:
        print(f"D2S auth contract smoke failed: {error}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"D2S auth contract smoke passed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
