"""Cost-model unit tests: threshold formula, savings sanity, decision optimality."""
import numpy as np
import pytest

from src.evaluation.cost import (baseline_cost, cost_threshold, evaluate_policy,
                                 total_cost)


def test_threshold_formula():
    # decline iff p*A > (1-p)*k*A  <=>  p > k/(1+k)
    assert cost_threshold(0.15) == pytest.approx(0.15 / 1.15)
    assert cost_threshold(1.0) == pytest.approx(0.5)
    for k in (0.05, 0.15, 0.3, 0.6, 1.0):
        t = cost_threshold(k)
        for amount in (1.0, 10.0, 12345.0):
            p_hi, p_lo = t + 1e-6, t - 1e-6
            assert p_hi * amount > (1 - p_hi) * k * amount
            assert p_lo * amount < (1 - p_lo) * k * amount


def test_threshold_is_amount_independent():
    k = 0.15
    t = cost_threshold(k)
    # two transactions with identical p and wildly different amounts must get
    # the same decision under expected-cost decisioning
    p = np.array([t + 0.01, t + 0.01])
    declined = p > t
    assert declined[0] == declined[1]


def test_perfect_classifier_saves_everything():
    y = np.array([0, 0, 1, 1, 0])
    amt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    declined = y == 1
    res = evaluate_policy(y, amt, declined, k=0.15)
    assert res["total_cost"] == 0.0
    assert res["savings"] == pytest.approx(1.0)


def test_approve_all_has_zero_savings_when_it_is_the_baseline():
    y = np.array([0] * 97 + [1] * 3)
    amt = np.full(100, 100.0)
    base = baseline_cost(y, amt, k=0.15)
    assert base["baseline_policy"] == "approve_all"  # 300 < 0.15*9700
    res = evaluate_policy(y, amt, np.zeros(100, bool), k=0.15)
    assert res["savings"] == pytest.approx(0.0)


def test_cost_accounting():
    y = np.array([1, 1, 0, 0])
    amt = np.array([100.0, 50.0, 200.0, 10.0])
    declined = np.array([True, False, True, False])
    # missed fraud: 50 ; false declines: 0.15 * 200
    assert total_cost(y, amt, declined, 0.15) == pytest.approx(50 + 30)
    res = evaluate_policy(y, amt, declined, 0.15)
    assert res["fraud_dollars_caught"] == pytest.approx(100.0)
    assert res["legit_dollars_declined"] == pytest.approx(200.0)
    assert res["tpr"] == pytest.approx(0.5)
    assert res["fpr"] == pytest.approx(0.5)


def test_analytic_threshold_is_cost_optimal_on_grid():
    """Brute-force check: on calibrated probabilities, no fixed threshold beats
    the analytic k/(1+k) threshold in expectation."""
    rng = np.random.RandomState(0)
    n = 200_000
    p = rng.beta(0.3, 6.0, n)            # calibrated by construction:
    y = (rng.rand(n) < p).astype(int)    # labels drawn from p
    amt = rng.lognormal(4.0, 1.0, n)
    k = 0.15
    t_star = cost_threshold(k)
    c_star = total_cost(y, amt, p > t_star, k)
    for t in np.linspace(0.01, 0.99, 60):
        c_t = total_cost(y, amt, p > t, k)
        assert c_star <= c_t * 1.02  # 2% tolerance for sampling noise
