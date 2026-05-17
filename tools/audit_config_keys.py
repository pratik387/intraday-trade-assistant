# tools/audit_config_keys.py
"""One-shot config audit: emit candidate-dead keys for human verification.

For each top-level functional key in config/configuration.json, scan all .py
and .json files under the runtime source directories for any literal reference
to that key string. Keys with zero hits are reported as CANDIDATE-DEAD and
must be human-verified before deletion (some may be referenced via dynamic
construction, eval, or external tooling).

Walks the tree in pure Python (no subprocess) so it runs in seconds rather
than minutes and works identically on Windows / Linux / Cygwin.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "configuration.json"

# Runtime code paths the audit considers. A key that does not appear under any
# of these is a candidate for removal from configuration.json.
SEARCH_DIRS = [
    "services",
    "structures",
    "pipelines",
    "gates",
    "tools",
    "oci",
    "broker",
    "config/pipelines",
]
SEARCH_FILES = ["main.py"]

SCANNED_EXTS = {".py", ".json"}

# Skip self and config-of-record so the key list does not match itself.
EXCLUDE_PATHS = {
    (ROOT / "tools" / "audit_config_keys.py").resolve(),
    CONFIG_PATH.resolve(),
}


def top_level_functional_keys() -> list[str]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [k for k in cfg.keys() if not k.startswith("_")]


def iter_source_files():
    for rel in SEARCH_DIRS:
        base = ROOT / rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in SCANNED_EXTS:
                continue
            if p.resolve() in EXCLUDE_PATHS:
                continue
            yield p
    for rel in SEARCH_FILES:
        p = ROOT / rel
        if p.exists() and p.resolve() not in EXCLUDE_PATHS:
            yield p


def main() -> None:
    keys = top_level_functional_keys()
    print(f"Auditing {len(keys)} functional top-level keys...\n")

    # Build hit-count per key by streaming each source file once.
    hit_files: dict[str, set[str]] = {k: set() for k in keys}
    quoted_keys = [(k, f'"{k}"', f"'{k}'") for k in keys]

    files_scanned = 0
    for path in iter_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  WARN: could not read {path}: {e}")
            continue
        files_scanned += 1
        rel = str(path.relative_to(ROOT))
        for key, dq, sq in quoted_keys:
            if dq in text or sq in text:
                hit_files[key].add(rel)

    print(f"Scanned {files_scanned} source files.\n")

    used = [(k, len(hit_files[k])) for k in keys if hit_files[k]]
    candidate_dead = [k for k in keys if not hit_files[k]]

    print(f"USED ({len(used)} keys):")
    for k, n in sorted(used):
        print(f"  {k} ({n} files)")

    print(f"\nCANDIDATE-DEAD ({len(candidate_dead)} keys):")
    for k in sorted(candidate_dead):
        print(f"  {k}")

    print(
        f"\nSUMMARY: USED={len(used)}  CANDIDATE-DEAD={len(candidate_dead)}  "
        f"TOTAL={len(keys)}"
    )
    print("--- NEXT: humans verify each candidate-dead key before deletion. ---")


if __name__ == "__main__":
    main()
