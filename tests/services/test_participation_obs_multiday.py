"""Multi-day participation is OBSERVED, never enforced — and why.

The overnight book measured a strong participation penalty in LIVE
(corr(participation, return) = -0.69; trades above 5% of a name's median daily
turnover produced 92% of that book's loss) which is ABSENT from its paper mirror
(corr -0.03). The mechanism is market impact, and idealised fills cannot contain
it.

The multi-day book is paper-only. The same test there can therefore only ask
whether thin names are worse TRADES, and the answer is no: corr(participation,
return) measures +0.067 / +0.076 / +0.148 across all / clean / ex-target_touch
cuts, and corr(log ADV, return) is about -0.06. Enforcing a cap on that evidence
would calibrate a control against data that structurally cannot contain the
thing being controlled.

Hence: log the distribution, enforce nothing, and let a live leg supply the
impact measurement if this book ever gets one.
"""
import ast
import inspect
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "services" / "execution" / "mtf_capitulation_handlers.py"
TEXT = SRC.read_text(encoding="utf-8")
RANKER = (Path(__file__).resolve().parents[2] / "services" / "cross_sectional_ranker.py").read_text(encoding="utf-8")


def test_adv_reaching_the_executor_is_point_in_time():
    """adv_prior is shifted one session, so it never contains the signal day.
    Using the unshifted `adv` would leak the trade day into its own denominator
    — the same look-ahead that inflated the 2026-08 study before correction."""
    assert '"adv_prior_inr"' in RANKER, "executor needs the SHIFTED adv"
    i = RANKER.index('df["adv_prior"]')
    assert "shift(1)" in RANKER[i:i + 200], "adv_prior must be shifted"
    # and the executor must read the shifted one, not the raw
    assert 'c.get("adv_prior_inr")' in TEXT
    assert 'c.get("adv_inr")' not in TEXT, "must not read the unshifted ADV"


def test_participation_is_logged():
    assert "PARTICIPATION_OBS" in TEXT


# the executable block, not the explanatory comment above it
_START = TEXT.index('_adv = c.get("adv_prior_inr")')
_BLOCK = TEXT[_START:TEXT.index("if qty <= 0:", _START)]


def test_participation_is_NOT_enforced():
    """The whole point. No qty may be reduced on the basis of participation."""
    assert "qty = " not in _BLOCK, "participation logging must not reassign qty"
    assert "continue" not in _BLOCK, "participation logging must not skip a trade"
    assert "sized.notional_inr =" not in _BLOCK, "must not rewrite the sized notional"


def test_logging_never_fires_on_a_rejected_trade():
    """qty<=0 trades are rejected upstream; logging them would pollute the sample."""
    assert "qty > 0" in _BLOCK, "must guard on a real position"


def test_module_still_imports_and_sizing_is_untouched():
    from services.execution import mtf_capitulation_handlers as H
    from services.risk.multiday_sizing import size_position
    assert callable(size_position)
    src = inspect.getsource(H)
    # the sizing call itself must still be driven by risk budget, not participation
    assert "size_position(" in src
    assert "participation" not in src.split("size_position(")[1][:400].lower(), \
        "sizing must not consult participation while it is observation-only"
