"""The token scripts take whatever the operator copies from the broker's redirect page.

2026-11-18: the morning Kite login pasted `<request_token>&action=login`. Kite rejected it,
token.txt kept the previous day's token, and paper and live both lost the day's only signal.
"""
import pytest

import broker.kite.get_access_token as kite_script
import broker.upstox.get_upstox_token as upstox_script
from broker.oauth_callback import extract_callback_param

KITE_TOKEN = "AbCdEf0123456789AbCdEf0123456789"
UPSTOX_CODE = "aB3xYz"
# The operator's real redirect URLs (2026-11-18), with made-up values.
KITE_REDIRECT = f"http://localhost:8000/api/kite-callback?action=login&type=login&status=success&request_token={KITE_TOKEN}"
UPSTOX_REDIRECT = f"http://localhost:8000/callback?code={UPSTOX_CODE}"


@pytest.mark.parametrize("pasted, param, want", [
    pytest.param(KITE_REDIRECT, "request_token", KITE_TOKEN, id="kite-full-url"),
    pytest.param(f"https://127.0.0.1/?request_token={KITE_TOKEN}&action=login&status=success",
                 "request_token", KITE_TOKEN, id="kite-token-first-in-query"),
    pytest.param(f"action=login&type=login&status=success&request_token={KITE_TOKEN}", "request_token", KITE_TOKEN,
                 id="kite-query-only"),
    pytest.param(f"{KITE_TOKEN}&action=login", "request_token", KITE_TOKEN, id="kite-token-plus-tail-2026-11-18"),
    pytest.param(KITE_TOKEN, "request_token", KITE_TOKEN, id="kite-bare-token"),
    pytest.param(f'  "{KITE_REDIRECT}"\n', "request_token", KITE_TOKEN, id="kite-quoted-with-spaces"),
    pytest.param(UPSTOX_REDIRECT, "code", UPSTOX_CODE, id="upstox-full-url"),
    pytest.param(f"localhost:8000/callback?code={UPSTOX_CODE}", "code", UPSTOX_CODE, id="upstox-url-without-scheme"),
    pytest.param(f"http://localhost:8000/callback?code={UPSTOX_CODE}#top", "code", UPSTOX_CODE, id="upstox-url-with-fragment"),
    pytest.param(UPSTOX_CODE, "code", UPSTOX_CODE, id="upstox-bare-code"),
])
def test_pulls_the_code_out_of_any_paste_shape(pasted, param, want):
    assert extract_callback_param(pasted, param) == want


@pytest.mark.parametrize("pasted, param", [
    pytest.param("", "request_token", id="empty"),
    pytest.param("  \n", "request_token", id="blank"),
    pytest.param("https://kite.zerodha.com/connect/login?v=3&api_key=abc123", "request_token",
                 id="login-url-instead-of-redirect"),
    pytest.param("http://localhost:8000/api/kite-callback?action=login&type=login&status=success", "request_token",
                 id="redirect-without-token"),
    pytest.param("http://localhost:8000/api/kite-callback", "request_token", id="url-cut-before-query"),
    pytest.param(UPSTOX_REDIRECT, "request_token", id="upstox-url-in-kite-script"),
    pytest.param(f"{KITE_TOKEN} {KITE_TOKEN}", "request_token", id="pasted-twice"),
])
def test_rejects_a_paste_without_a_usable_code(pasted, param):
    with pytest.raises(ValueError):
        extract_callback_param(pasted, param)


def _run_main(script, monkeypatch, pasted, via):
    if via == "flag":
        monkeypatch.setattr("sys.argv", ["prog", "--auth-code", pasted])
    else:
        monkeypatch.setattr("sys.argv", ["prog"])
        monkeypatch.setattr("builtins.input", lambda prompt="": pasted)
    return script.main()


@pytest.fixture
def kite_exchanges(monkeypatch):
    """Request tokens main() hands to the exchange; stubbed, so no network call and no token.txt write."""
    seen = []
    monkeypatch.setattr(kite_script.env, "KITE_API_KEY", "kite_key")
    monkeypatch.setattr(kite_script.env, "KITE_API_SECRET", "kite_secret")
    monkeypatch.setattr(kite_script, "_exchange", lambda key, secret, token: seen.append(token) or 0)
    return seen


@pytest.fixture
def upstox_exchanges(monkeypatch):
    """Auth codes main() hands to the exchange; stubbed, so no network call, browser or token.txt write."""
    seen = []
    monkeypatch.setattr(upstox_script.env, "UPSTOX_API_KEY", "upstox_key_1234")
    monkeypatch.setattr(upstox_script.env, "UPSTOX_API_SECRET", "upstox_secret")
    monkeypatch.setattr(upstox_script.env, "UPSTOX_REDIRECT_URI", "http://localhost:8000/callback")
    monkeypatch.setattr(upstox_script.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(upstox_script, "_exchange_code", lambda key, secret, redirect, code: seen.append(code) or 0)
    return seen


@pytest.mark.parametrize("via", ["prompt", "flag"])
def test_kite_script_exchanges_the_token_from_a_pasted_redirect_url(kite_exchanges, monkeypatch, via):
    assert _run_main(kite_script, monkeypatch, KITE_REDIRECT, via) == 0
    assert kite_exchanges == [KITE_TOKEN]


@pytest.mark.parametrize("via", ["prompt", "flag"])
def test_upstox_script_exchanges_the_code_from_a_pasted_redirect_url(upstox_exchanges, monkeypatch, via):
    assert _run_main(upstox_script, monkeypatch, UPSTOX_REDIRECT, via) == 0
    assert upstox_exchanges == [UPSTOX_CODE]


def test_kite_script_stops_before_the_exchange_when_the_paste_has_no_token(kite_exchanges, monkeypatch):
    assert _run_main(kite_script, monkeypatch, "http://localhost:8000/api/kite-callback?action=login&status=success",
                     "prompt") == 1
    assert kite_exchanges == []


def test_upstox_script_stops_before_the_exchange_when_the_paste_has_no_code(upstox_exchanges, monkeypatch):
    assert _run_main(upstox_script, monkeypatch, "http://localhost:8000/callback?error=access_denied", "prompt") == 1
    assert upstox_exchanges == []
