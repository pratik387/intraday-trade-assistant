"""
Kite Access Token Generator

Exchanges request_token for access_token.

Usage (interactive):
    python broker/kite/get_access_token.py

Usage (print login URL and exit):
    python broker/kite/get_access_token.py --print-url

Usage (non-interactive):
    python broker/kite/get_access_token.py --auth-code <request_token>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.env_setup import env


def _build_login_url(api_key: str) -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def _exchange(api_key: str, api_secret: str, request_token: str) -> int:
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        print(f"[ERROR] Failed to generate access token: {e}")
        return 2
    access_token = data["access_token"]
    token_file = Path(__file__).parent / "token.txt"
    token_file.write_text(access_token)
    print(f"[SAVED] Token saved to: {token_file}")
    try:
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"[VERIFIED] User: {profile.get('user_name', 'Unknown')}")
    except Exception as e:
        print(f"[WARN] Token verify failed: {e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kite access token generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print-url", action="store_true", help="Print login URL and exit")
    group.add_argument("--auth-code", help="Exchange this request_token non-interactively")
    args = parser.parse_args()

    api_key = env.KITE_API_KEY
    api_secret = env.KITE_API_SECRET
    if not api_key or not api_secret:
        print("[ERROR] KITE_API_KEY or KITE_API_SECRET not set in .env file")
        return 1

    if args.print_url:
        print(_build_login_url(api_key))
        return 0

    if args.auth_code:
        return _exchange(api_key, api_secret, args.auth_code.strip())

    # Interactive fallback (original behavior).
    print("=" * 70)
    print("Kite Access Token Generator")
    print("=" * 70)
    print(f"Login URL: {_build_login_url(api_key)}")
    print("After logging in, you'll be redirected to a URL with request_token=XXXXX")
    request_token = input("Paste the request_token from the URL: ").strip()
    if not request_token:
        print("[ERROR] No request_token provided")
        return 1
    return _exchange(api_key, api_secret, request_token)


if __name__ == "__main__":
    sys.exit(main())
