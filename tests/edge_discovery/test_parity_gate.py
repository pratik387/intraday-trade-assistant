from tools.edge_discovery.validation.parity_gate import compare_parity, ParityTolerance


def _tol() -> ParityTolerance:
    return ParityTolerance(pf_pct=10.0, wr_pp=5.0, n_pct=10.0)


def test_parity_passes_when_within_tolerance():
    live = {"pf": 1.36, "wr": 0.70, "n": 797}
    framework = {"pf": 1.30, "wr": 0.69, "n": 780}
    verdict = compare_parity(framework, live, _tol())
    assert verdict.passed is True


def test_parity_fails_when_pf_outside_tolerance():
    live = {"pf": 1.36, "wr": 0.70, "n": 797}
    framework = {"pf": 1.05, "wr": 0.70, "n": 797}  # PF 23% below
    verdict = compare_parity(framework, live, _tol())
    assert verdict.passed is False
    assert any("pf" in r for r in verdict.failures)


def test_parity_fails_when_wr_outside_tolerance():
    live = {"pf": 1.36, "wr": 0.70, "n": 797}
    framework = {"pf": 1.36, "wr": 0.62, "n": 797}  # WR 8pp below
    verdict = compare_parity(framework, live, _tol())
    assert verdict.passed is False


def test_parity_fails_when_n_outside_tolerance():
    live = {"pf": 1.36, "wr": 0.70, "n": 797}
    framework = {"pf": 1.36, "wr": 0.70, "n": 600}  # N 25% below
    verdict = compare_parity(framework, live, _tol())
    assert verdict.passed is False
