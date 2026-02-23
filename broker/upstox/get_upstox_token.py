"""
Upstox OAuth2 Access Token Generator

Exchanges authorization code for access token via Upstox OAuth2 flow.
Run this daily before trading session (Upstox tokens expire at midnight).

Usage:
    python broker/upstox/get_upstox_token.py

Flow:
    1. Opens Upstox login page in browser
    2. User logs in → redirected with auth code in URL
    3. Script exchanges code for access token
    4. Saves to broker/upstox/token.txt
"""

import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.env_setup import env


def get_upstox_token():
    """Exchange authorization code for Upstox access token."""

    print("=" * 70)
    print("Upstox Access Token Generator")
    print("=" * 70)
    print()

    api_key = getattr(env, "UPSTOX_API_KEY", None)
    api_secret = getattr(env, "UPSTOX_API_SECRET", None)
    redirect_uri = getattr(env, "UPSTOX_REDIRECT_URI", None) or "http://localhost:8000/callback"

    if not api_key:
        print("[ERROR] UPSTOX_API_KEY not set in .env file")
        print("  Add: UPSTOX_API_KEY=your_api_key")
        return

    if not api_secret:
        print("[ERROR] UPSTOX_API_SECRET not set in .env file")
        print("  Add: UPSTOX_API_SECRET=your_api_secret")
        return

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Redirect URI: {redirect_uri}")
    print()

    # Step 1: Open browser for user login
    auth_url = "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode({
        "client_id": api_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    })

    print("Opening Upstox login page in browser...")
    print(f"  URL: {auth_url}")
    print()

    try:
        webbrowser.open(auth_url)
    except Exception:
        print("  [WARN] Could not open browser. Copy the URL above manually.")

    print("After logging in, you'll be redirected to a URL like:")
    print(f"  {redirect_uri}?code=XXXXX")
    print()

    auth_code = input("Paste the 'code' from the redirect URL: ").strip()

    if not auth_code:
        print("[ERROR] No authorization code provided")
        return

    # Step 2: Exchange code for token
    print()
    print("Exchanging authorization code for access token...")

    try:
        import requests

        token_url = "https://api.upstox.com/v2/login/authorization/token"
        resp = requests.post(
            token_url,
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
            print(f"[ERROR] HTTP {resp.status_code}")
            print(f"  Response: {resp.text}")
            print()
            print("Debug info:")
            print(f"  client_id: {api_key[:8]}...{api_key[-4:]}")
            print(f"  client_secret: {api_secret[:3]}...{api_secret[-3:]}")
            print(f"  redirect_uri: {redirect_uri}")
            print(f"  code: {auth_code}")
            return

        data = resp.json()
        access_token = data.get("access_token")

        if not access_token:
            print(f"[ERROR] No access_token in response: {data}")
            return

        print()
        print("[SUCCESS] Access token generated!")
        print(f"  Token: {access_token[:20]}...{access_token[-10:]}")
        print()

        # Save to token.txt
        token_file = Path(__file__).parent / "token.txt"
        token_file.write_text(access_token)
        print(f"[SAVED] Token saved to: {token_file}")
        print()

        # Verify token works
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
            print("[VERIFIED] Token is valid!")
            print(f"  User: {profile.get('user_name', 'Unknown')}")
            print(f"  Email: {profile.get('email', 'Unknown')}")
        else:
            print(f"[WARN] Token verification returned {verify_resp.status_code}")

        print()
        print("=" * 70)
        print("Ready! Start with:")
        print("  python main.py --paper-trading --data-source upstox")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Failed to exchange code: {e}")
        print()
        print("Common issues:")
        print("  1. Authorization code already used (get a new one)")
        print("  2. Wrong API secret")
        print("  3. Code expired (valid for ~5 minutes)")
        print("  4. Redirect URI mismatch with Upstox app settings")


if __name__ == "__main__":
    get_upstox_token()
