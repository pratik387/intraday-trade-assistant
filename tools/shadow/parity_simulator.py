"""Parity simulator — sub-project #4.

Replays a cached gate_input.jsonl through services.gate_chain.LiveGateChain
and writes the admit decisions to a CSV. Bit-exact with live's gate output by
construction (wraps the same Python module).

Usage:
    python tools/shadow/parity_simulator.py \\
        --gate-input <path-to-gate_input.jsonl OR dir-of-session-dirs> \\
        --config <path-to-configuration.json> \\
        --output <path-to-sim_admits.csv>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate-input", required=True,
                   help="Path to gate_input.jsonl OR directory of session subfolders")
    p.add_argument("--config", required=True,
                   help="Path to configuration.json")
    p.add_argument("--output", required=True,
                   help="Path to write sim_admits.csv")
    args = p.parse_args()
    raise NotImplementedError("Task 5 implements the replay loop")


if __name__ == "__main__":
    main()
