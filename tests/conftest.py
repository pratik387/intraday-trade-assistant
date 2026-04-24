"""Pytest configuration to add project root to path."""
import sys
from pathlib import Path

# Add project root to path immediately when conftest is imported
project_root = str(Path(__file__).parent.parent)

# Ensure project root is at position 0
while project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)
