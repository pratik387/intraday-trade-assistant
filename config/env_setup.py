# @role: Environment loader for backend settings
# @used_by: kite_client.py, tick_listener.py

# @filter_type: utility
# @tags: env, config, bootstrap
# config/env.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Determine project root & ENV ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ENV  = os.getenv("ENV", "development").lower()

# ─── Load the right .env file ────────────────────────────────────────────────
env_path = ROOT / f".env.{ENV}"
if not env_path.exists():
    raise FileNotFoundError(f"Missing environment file: {env_path.name}")
load_dotenv(env_path)

# ─── Helpers to read token from file ─────────────────────────────────────────
def _read_token_file():
    """Read Kite access token from broker/kite/token.txt (if exists)"""
    token_file = ROOT / "broker" / "kite" / "token.txt"
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                return token
        except Exception:
            pass
    return None


def _read_upstox_token_file():
    """Read Upstox access token from broker/upstox/token.txt (if exists)"""
    token_file = ROOT / "broker" / "upstox" / "token.txt"
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                return token
        except Exception:
            pass
    return None

# ─── Expose your environment settings ────────────────────────────────────────
class EnvConfig:
    ENV                = ENV
    KITE_REDIRECT_URI  = os.getenv("KITE_REDIRECT_URI")
    KITE_API_KEY       = os.getenv("KITE_API_KEY")
    KITE_API_SECRET    = os.getenv("KITE_API_SECRET")

    # Priority: token.txt > env variable
    KITE_ACCESS_TOKEN = _read_token_file() or os.getenv("KITE_ACCESS_TOKEN")
    DRY_RUN           = os.getenv("DRY_RUN")

    # Upstox credentials (for hybrid data feed mode)
    UPSTOX_API_KEY       = os.getenv("UPSTOX_API_KEY")
    UPSTOX_API_SECRET    = os.getenv("UPSTOX_API_SECRET")
    UPSTOX_REDIRECT_URI  = os.getenv("UPSTOX_REDIRECT_URI")
    UPSTOX_ACCESS_TOKEN  = _read_upstox_token_file() or os.getenv("UPSTOX_ACCESS_TOKEN")

    @classmethod
    def validate_for_paper_trading(cls):
        """Validate required credentials for paper trading mode"""
        errors = []

        if not cls.KITE_API_KEY:
            errors.append("KITE_API_KEY not set in environment")

        if not cls.KITE_ACCESS_TOKEN:
            errors.append("KITE_ACCESS_TOKEN not set - run auth flow first")

        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Paper trading requires valid credentials:\n{error_msg}")

        print("[OK] Environment validation passed for paper trading")

env = EnvConfig()