import numpy as np
import pytest

from quant_research.pricing import black76
from quant_research.pricing.svi import SVIParams, durrleman_g, fit_svi, synthetic_chain


def test_hull_reference_value():
    # Hull, Options Futures & Other Derivatives: European put on futures,
    # F=20, K=20, r=9%, T=4/12, sigma=25% -> 1.12
    p = black76.price(20, 20, 4 / 12, 0.09, 0.25, is_call=False)
    assert float(p) == pytest.approx(1.12, abs=0.005)


def test_put_call_parity():
    F, K, T, r, s = 21000, 20500, 0.25, 0.045, 0.22
    c = black76.price(F, K, T, r, s, True)
    p = black76.price(F, K, T, r, s, False)
    assert float(c - p) == pytest.approx(np.exp(-r * T) * (F - K), rel=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    F, K, T, r, s = 21000.0, 21250.0, 0.1, 0.04, 0.2
    g = black76.greeks(F, K, T, r, s, is_call)
    h = 1.0

    def px(**kw):
        args = {"F": F, "K": K, "T": T, "r": r, "sigma": s, **kw}
        return float(
            black76.price(args["F"], args["K"], args["T"], args["r"], args["sigma"], is_call)
        )

    assert float(g.delta) == pytest.approx((px(F=F + h) - px(F=F - h)) / (2 * h), rel=1e-5)
    assert float(g.gamma) == pytest.approx((px(F=F + h) - 2 * px() + px(F=F - h)) / h**2, rel=1e-3)
    assert float(g.vega) == pytest.approx(
        (px(sigma=s + 1e-4) - px(sigma=s - 1e-4)) / 2e-4, rel=1e-5
    )
    dt = 1e-5
    assert float(g.theta) == pytest.approx(-(px(T=T + dt) - px(T=T - dt)) / (2 * dt), rel=1e-4)


def test_implied_vol_round_trip_and_bounds():
    for is_call in (True, False):
        for s in (0.08, 0.2, 0.6):
            p = float(black76.price(21000, 20000, 0.2, 0.04, s, is_call))
            assert black76.implied_vol(p, 21000, 20000, 0.2, 0.04, is_call) == pytest.approx(
                s, abs=1e-7
            )
    assert np.isnan(black76.implied_vol(-1.0, 100, 100, 1, 0))
    assert np.isnan(black76.implied_vol(200.0, 100, 100, 1, 0))


def test_svi_recovers_known_smile():
    true = SVIParams(a=0.002, b=0.04, rho=-0.7, m=0.02, sigma=0.08)
    F, T, r = 21000.0, 30 / 365, 0.045
    chain = synthetic_chain(F, T, r, true, noise_vol=0.001, seed=4)
    iv = np.array(
        [
            black76.implied_vol(p, F, k, T, r, bool(c))
            for p, k, c in zip(chain["price"], chain["strike"], chain["is_call"], strict=True)
        ]
    )
    fit = fit_svi(chain["log_moneyness"], iv, T)
    k = chain["log_moneyness"]
    assert fit.rmse_vol < 0.002
    assert np.max(np.abs(fit.params.implied_vol(k, T) - true.implied_vol(k, T))) < 0.004
    assert fit.butterfly_arbitrage_free


def test_durrleman_detects_arbitrage():
    bad = SVIParams(a=-0.05, b=2.0, rho=0.0, m=0.0, sigma=0.01)
    assert (durrleman_g(bad, np.linspace(-0.2, 0.2, 51)) < 0).any()


def test_svi_needs_enough_points():
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1], [0.2, 0.2], 0.1)
