# config/filters_setup.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY_CONFIG_PATH = ROOT / "configuration.json"

_CFG = None  # memoized merged config

def _load_json(path: Path, required: bool) -> dict:
    """Load JSON safely. If required and missing, raise; else return {}."""
    if not path.exists():
        msg = f"[config] file not found: {path.name}"
        if required:
            raise FileNotFoundError(f"{path} not found")
        else: 
            print(msg)
            return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        
        # bubble up: no defaults here, as per project policy
        raise

def load_filters(force_reload: bool = False) -> dict:
    """
    Load entry_config.json and (optionally) overlay exit_config.json.
    No global REQUIRED_KEYS validation; each module should check what it needs.
    """
    global _CFG
    if _CFG is not None and not force_reload:
        return _CFG

    entry = _load_json(ENTRY_CONFIG_PATH, required=True)

    cfg = {}
    cfg.update(entry)
    _CFG = cfg
    return cfg

# IMPORTANT: do NOT auto-load at import time (avoids side effects & makes testing easier).
# Remove any global 'filters = load_filters()' pattern.
