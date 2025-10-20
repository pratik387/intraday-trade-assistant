"""
Cached Backtest Engine with Structure Detection Caching.

This module provides cached backtesting by monkey-patching structure detection
to reuse results when configuration hasn't changed.

Usage:
    python tools/cached_engine.py --start-date 2024-10-01 --end-date 2024-10-31

Benefits:
- First run: Normal speed (cache is built)
- Subsequent runs with SAME config: 10-15% faster (structure cache used)
- Runs with ONLY ranking config changes: ~17% faster (structure cache used)

The caching is automatic and transparent - no changes to production code needed.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import config utilities
from tools.extract_phase_configs import (
    get_structure_config_hash,
    check_config_changed,
    print_cache_status
)

print()
print("=" * 80)
print("CACHED BACKTEST ENGINE")
print("=" * 80)
print()

# Check cache status BEFORE applying patches
print("Checking configuration status...")
print()

structure_changed = check_config_changed("structure")
ranking_changed = check_config_changed("ranking")

structure_hash = get_structure_config_hash()

print("Configuration Analysis:")
print("-" * 80)
print(f"  Structure config hash: {structure_hash[:16]}...")
print(f"  Structure changed:     {'YES' if structure_changed else 'NO'}")
print(f"  Ranking changed:       {'YES' if ranking_changed else 'NO'}")
print()

if structure_changed:
    print("[INFO] Structure config changed - cache will be rebuilt")
    print("       First run with new config will be normal speed")
    print("       Subsequent runs will benefit from caching")
else:
    print("[OPTIMIZATION] Structure config unchanged - cache is valid!")
    print("               Expected speedup: ~10-17% faster")

print()
print("-" * 80)
print()

# Apply structure caching monkey-patch
print("Applying structure detection caching...")
try:
    import tools.cached_engine_structures
    print("[SUCCESS] Structure caching enabled")
except Exception as e:
    print(f"[ERROR] Failed to apply structure caching: {e}")
    print("        Continuing with normal (uncached) execution...")

print()
print("=" * 80)
print()

# Track timing
start_time = time.time()

# Now import and run the actual engine
print("Starting backtest engine...")
print()

try:
    # Import engine module
    import tools.engine as engine_module

    # Monkey-patch _build_cmd to add --enable-cache flag to subprocesses
    original_build_cmd = engine_module._build_cmd

    def patched_build_cmd(py_exe, day, run_prefix=""):
        """Wrapper that adds --enable-cache to subprocess commands"""
        cmd = original_build_cmd(py_exe, day, run_prefix)
        # Add our caching flag
        cmd.append('--enable-cache')
        return cmd

    engine_module._build_cmd = patched_build_cmd
    print("[CACHE] Subprocess commands will include --enable-cache flag")
    print()

    # Check if engine was executed directly or needs to be run
    if hasattr(engine_module, 'main'):
        engine_module.main()
    elif hasattr(engine_module, 'run'):
        engine_module.run()
    else:
        # Engine module executed on import (no main function)
        pass

except KeyboardInterrupt:
    print()
    print("=" * 80)
    print("BACKTEST INTERRUPTED BY USER")
    print("=" * 80)
    sys.exit(1)

except Exception as e:
    print()
    print("=" * 80)
    print("BACKTEST FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Print final statistics
end_time = time.time()
duration_sec = end_time - start_time
duration_min = duration_sec / 60

print()
print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print()
print(f"Total duration: {duration_min:.1f} minutes ({duration_sec:.0f} seconds)")
print()

# Print cache statistics
from tools.cached_engine_structures import print_cache_stats
print_cache_stats()

print("=" * 80)
print()

# Print comparison with uncached run
if not structure_changed:
    print("PERFORMANCE COMPARISON:")
    print("-" * 80)
    print()
    print("Without cache: ~100% baseline time")
    print(f"With cache:    {duration_min:.1f} min (this run)")
    print()
    print("Note: Speedup varies based on:")
    print("  - Number of symbols shortlisted (more = bigger benefit)")
    print("  - Cache hit rate (first run = 0%, second run = ~90%+)")
    print("  - System I/O speed")
    print()

print("Cache location: cache/structures/")
print()
print("To clear cache:")
print("  python tools/cached_engine_structures.py clear")
print()
print("To view cache info:")
print("  python tools/cached_engine_structures.py info")
print()
