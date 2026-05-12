"""
Upstox OAuth2 Access Token Generator

Exchanges authorization code for access token via Upstox OAuth2 flow.
Run this daily before trading session (Upstox tokens expire at midnight).

Usage (interactive):
    python broker/upstox/get_upstox_token.py

Usage (print login URL and exit):
    python broker/upstox/get_upstox_token.py --print-url

Usage (non-interactive):
    python broker/upstox/get_upstox_token.py --auth-code <code>
"""

import argparse
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.env_setup import env


def _build_auth_url(api_key: str, redirect_uri: str) -> str:
    return "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode({
        "client_id": api_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    })


def _exchange_code(api_key: str, api_secret: str, redirect_uri: str, auth_code: str) -> int:
    import requests

    print("Exchanging authorization code for access token...")
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={
            "code": auth_code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
        return 2

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        print(f"[ERROR] No access_token in response: {data}")
        return 3

    token_file = Path(__file__).parent / "token.txt"
    token_file.write_text(access_token)
    print(f"[SAVED] Token saved to: {token_file}")

    verify_resp = requests.get(
        "https://api.upstox.com/v2/user/profile",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        timeout=10,
    )
    if verify_resp.status_code == 200:
        profile = verify_resp.json().get("data", {})
        print(f"[VERIFIED] User: {profile.get('user_name', 'Unknown')}")
    else:
        print(f"[WARN] Token verification returned {verify_resp.status_code}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Upstox OAuth2 token generator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print-url", action="store_true", help="Print login URL and exit")
    group.add_argument("--auth-code", help="Exchange this authorization code non-interactively")
    args = parser.parse_args()

    api_key = getattr(env, "UPSTOX_API_KEY", None)
    api_secret = getattr(env, "UPSTOX_API_SECRET", None)
    redirect_uri = getattr(env, "UPSTOX_REDIRECT_URI", None) or "http://localhost:8000/callback"

    if not api_key or not api_secret:
        print("[ERROR] UPSTOX_API_KEY or UPSTOX_API_SECRET not set in .env file")
        return 1

    auth_url = _build_auth_url(api_key, redirect_uri)

    if args.print_url:
        print(auth_url)
        return 0

    if args.auth_code:
        return _exchange_code(api_key, api_secret, redirect_uri, args.auth_code.strip())

    # Interactive fallback (original behavior).
    print("=" * 70)
    print("Upstox Access Token Generator")
    print("=" * 70)
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Opening: {auth_url}")
    try:
        webbrowser.open(auth_url)
    except Exception:
        print("[WARN] Could not open browser. Copy the URL above manually.")
    print(f"After logging in, you'll be redirected to: {redirect_uri}?code=XXXXX")
    auth_code = input("Paste the 'code' from the redirect URL: ").strip()
    if not auth_code:
        print("[ERROR] No authorization code provided")
        return 1
    return _exchange_code(api_key, api_secret, redirect_uri, auth_code)


if __name__ == "__main__":
    sys.exit(main())
