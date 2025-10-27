"""
Simple script to exchange request_token for access_token
Run this after you get the request_token from Kite redirect URL.

Usage:
    python broker/kite/get_access_token.py
"""

import sys
from pathlib import Path

# Add project root to path (go up 2 levels: broker/kite -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.env_setup import env

def get_access_token():
    """Exchange request_token for access_token"""

    print("=" * 70)
    print("Kite Access Token Generator")
    print("=" * 70)
    print()

    # Check if API key and secret are set
    api_key = env.KITE_API_KEY
    api_secret = env.KITE_API_SECRET

    if not api_key:
        print("[ERROR] KITE_API_KEY not set in .env file")
        print()
        print("Please add to your .env file:")
        print("  KITE_API_KEY=your_api_key")
        return

    if not api_secret:
        print("[ERROR] KITE_API_SECRET not set in .env file")
        print()
        print("Please add to your .env file:")
        print("  KITE_API_SECRET=your_api_secret")
        return

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print()

    # Get request token from user
    print("After logging in to Kite, you'll be redirected to a URL like:")
    print("  http://localhost:8000/api/kite-callback?request_token=XXXXX&...")
    print()
    request_token = input("Paste the request_token from the URL: ").strip()

    if not request_token:
        print("[ERROR] No request_token provided")
        return

    print()
    print("Exchanging request_token for access_token...")
    print()

    try:
        from kiteconnect import KiteConnect

        # Initialize Kite
        kite = KiteConnect(api_key=api_key)

        # Generate session
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]

        print("[SUCCESS] Access token generated!")
        print()
        print(f"Access Token: {access_token}")
        print()

        # Save to token.txt
        token_file = Path(__file__).parent / "token.txt"
        token_file.write_text(access_token)

        print(f"[SAVED] Token saved to: {token_file}")
        print()

        # Verify token works
        kite.set_access_token(access_token)
        profile = kite.profile()

        print("[VERIFIED] Token is valid!")
        print(f"  User: {profile.get('user_name', 'Unknown')}")
        print(f"  Email: {profile.get('email', 'Unknown')}")
        print()

        print("=" * 70)
        print("You're ready to trade! Start with:")
        print("  python main.py --paper-trading")
        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Failed to generate access token: {e}")
        print()
        print("Common issues:")
        print("  1. Request token already used (get a new one)")
        print("  2. Wrong API secret")
        print("  3. Request token expired (valid for ~5 minutes)")
        print()
        print("Solution: Login to Kite again and get a fresh request_token")


if __name__ == "__main__":
    get_access_token()
