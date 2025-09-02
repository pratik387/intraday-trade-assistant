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

# ─── Expose your environment settings ────────────────────────────────────────
class EnvConfig:
    ENV                = ENV
    KITE_REDIRECT_URI  = os.getenv("KITE_REDIRECT_URI")
    KITE_API_KEY       = os.getenv("KITE_API_KEY")
    KITE_API_SECRET    = os.getenv("KITE_API_SECRET")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
    DRY_RUN           = os.getenv("DRY_RUN")

env = EnvConfig()