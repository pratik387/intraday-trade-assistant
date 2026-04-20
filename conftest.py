"""Root conftest.py for pytest.

Patches io.text_encoding so that Path.read_text(encoding=None) and open()
default to UTF-8 on all platforms. On Windows, the default is cp1252 which
cannot encode Unicode characters like ≥ (U+2265) and — (U+2014) that appear
in stage report markdown files.

This is a no-op on Linux/macOS (where UTF-8 is already the default) and on
Python 3.15+ (where UTF-8 mode will be the default). It is equivalent to
running with PYTHONUTF8=1 but can be applied after interpreter startup by
patching io.text_encoding.
"""
import io as _io

_orig_text_encoding = _io.text_encoding


def _utf8_text_encoding(encoding, stacklevel=2):
    if encoding is None:
        return "utf-8"
    return _orig_text_encoding(encoding, stacklevel + 1)


_io.text_encoding = _utf8_text_encoding
