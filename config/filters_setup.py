# config/filters_setup.py
import json
from pathlib import Path
from config.logging_config import get_loggers

logger, trade_logger = get_loggers()

ROOT = Path(__file__).resolve().parent
ENTRY_CONFIG_PATH = ROOT / "entry_config.json"
EXIT_CONFIG_PATH  = ROOT / "exit_config.json"

_CFG = None  # memoized merged config

def _load_json(path: Path, required: bool) -> dict:
    """Load JSON safely. If required and missing, raise; else return {}."""
    if not path.exists():
        msg = f"[config] file not found: {path.name}"
        if required:
            logger.error(msg)
            raise FileNotFoundError(f"{path} not found")
        else:
            logger.warning(msg)
            return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"[config] failed to load {path.name}: {e}")
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
    exit_ = _load_json(EXIT_CONFIG_PATH, required=False)  # optional overlay

    cfg = {}
    cfg.update(entry)
    if exit_:
        cfg.update(exit_)

    logger.info(f"[config] loaded: entry={'yes' if entry else 'no'}, exit={'yes' if exit_ else 'no'}, keys={len(cfg)}")
    _CFG = cfg
    return cfg

# IMPORTANT: do NOT auto-load at import time (avoids side effects & makes testing easier).
# Remove any global 'filters = load_filters()' pattern.
