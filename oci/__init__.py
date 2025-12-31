# OCI integration tools
#
# This local package shadows the OCI SDK when the project root is on sys.path.
# We detect this and proxy to the real OCI SDK from site-packages.

import sys
from pathlib import Path

# Check if we're the shadow package (not in site-packages)
_this_dir = Path(__file__).parent
_is_shadow = 'site-packages' not in str(_this_dir) and 'dist-packages' not in str(_this_dir)

if _is_shadow:
    # Find and load the real OCI SDK
    _real_oci = None
    _project_root = str(_this_dir.parent)

    # Temporarily remove project root from sys.path to find real OCI
    _saved_path = sys.path.copy()
    sys.path = [p for p in sys.path if _project_root not in p]

    # Also remove any cached oci modules
    _cached_modules = {k: v for k, v in sys.modules.items() if k == 'oci' or k.startswith('oci.')}
    for k in _cached_modules:
        del sys.modules[k]

    try:
        import oci as _real_oci
    except ImportError:
        _real_oci = None
    finally:
        # Restore sys.path
        sys.path = _saved_path

    if _real_oci:
        # Re-export all public attributes from the real OCI SDK
        for _attr in dir(_real_oci):
            if not _attr.startswith('_'):
                globals()[_attr] = getattr(_real_oci, _attr)

        # Also register the real module in sys.modules for submodule imports
        sys.modules['oci'] = _real_oci
    else:
        # OCI SDK not installed
        class _OCINotInstalled:
            def __getattr__(self, name):
                raise ImportError(
                    "OCI SDK is not installed. Install it with: pip install oci\n"
                    "This is required for OCI cloud operations."
                )

        config = _OCINotInstalled()
        object_storage = _OCINotInstalled()
        container_engine = _OCINotInstalled()
