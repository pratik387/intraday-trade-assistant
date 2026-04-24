"""Pytest configuration to add project root to path."""
import sys
from pathlib import Path

# Add project root to path immediately when conftest is imported
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
