"""Single source of truth for interpreting a setup's ``cb_state``.

ALLOWLIST semantics: a setup may open NEW positions only when its ``cb_state``
is one of ``ACTIVE_CB_STATES``. Any other value — including typos, hand-edited
config values, and states added by future monitors — blocks entries.

Review finding 2026-07-28: the previous checks were independently-maintained
BLOCKLISTS in three files (``plan_orchestrator``, ``overnight_handlers``,
``mtf_capitulation_handlers``); an unknown cb_state value fell through and
fired at full size. Every entry gate must call :func:`is_cb_active` instead of
comparing strings locally.

A missing ``cb_state`` key defaults to ``"enabled"`` — setups are active until
a breaker/monitor writes a state (jobs/check_circuit_breakers.py writes
``"disabled"``; jobs/check_edge_integrity.py writes ``"paused_precondition"``).
Exit paths never consult cb_state: existing positions always run their exits.
"""

ACTIVE_CB_STATES = ("enabled", "forward_validation")


def is_cb_active(setup_cfg: dict) -> bool:
    """True iff this setup's cb_state permits opening NEW positions."""
    return setup_cfg.get("cb_state", "enabled") in ACTIVE_CB_STATES
