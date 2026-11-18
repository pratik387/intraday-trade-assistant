"""Pull the one-time login code out of whatever the operator copied from a broker's redirect page.

After login, Kite redirects to `<redirect>?request_token=...&action=login&status=success` and Upstox to
`<redirect>?code=...&state=...`. The token scripts accept the whole URL, its query string, the code
followed by the parameters after it (`<code>&action=login`), or the bare code.
"""
import re

_CODE_CHARS = re.compile(r"[A-Za-z0-9_-]+")


def extract_callback_param(pasted: str, param: str) -> str:
    """Return the value of `param` from a pasted redirect URL, its query string, or the bare value.

    Raises ValueError when the paste holds no usable value. The message never repeats the paste.
    """
    text = pasted.strip().strip("\"'").strip()
    if not text:
        raise ValueError(f"nothing pasted; paste the redirect URL or the {param}")
    query = text.split("?", 1)[1] if "?" in text else text
    head, *rest = query.split("#", 1)[0].split("&")
    pairs = dict(part.split("=", 1) for part in (head, *rest) if "=" in part)
    if param in pairs:
        value = pairs[param]
    elif "=" not in head:
        value = head
    else:
        raise ValueError(f"no {param}= in the pasted text; copy the URL the browser lands on after login")
    if not _CODE_CHARS.fullmatch(value):
        raise ValueError(f"the pasted text has no usable {param}; paste the whole redirect URL from the browser bar")
    return value
