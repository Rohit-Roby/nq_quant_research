"""Raw SVI implied-volatility smile fitting (Gatheral, 2004).

Total implied variance as a function of log-moneyness ``k = ln(K/F)``::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Constraints enforced in the fit: ``b >= 0``, ``|rho| < 1``, ``sigma > 0`` and
``a + b * sigma * sqrt(1 - rho^2) >= 0`` (non-negative minimum variance). A
Durrleman butterfly-arbitrage check is provided for the fitted slice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from quant_research.pricing import black76


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: ArrayLike) -> NDArray[np.float64]:
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: ArrayLike, T: float) -> NDArray[np.float64]:
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    T: float
    rmse_vol: float
    n_points: int
    butterfly_arbitrage_free: bool


def durrleman_g(p: SVIParams, k: ArrayLike) -> NDArray[np.float64]:
    """Durrleman's condition g(k) >= 0 <=> no butterfly arbitrage in the slice."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    root = np.sqrt(x**2 + p.sigma**2)
    w = p.total_variance(k)
    w1 = p.b * (p.rho + x / root)
    w2 = p.b * p.sigma**2 / root**3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_svi(k: ArrayLike, iv: ArrayLike, T: float, weights: ArrayLike | None = None) -> SVIFit:
    """Fit raw SVI to a single expiry slice of (log-moneyness, implied vol)."""
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    ok = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[ok], iv[ok]
    if len(k) < 5:
        raise ValueError("need at least 5 valid quotes to fit SVI")
    w_mkt = iv**2 * T
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)[ok]

    def unpack(x: NDArray[np.float64]) -> SVIParams:
        return SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])

    def resid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        p = unpack(x)
        penalty = max(0.0, -(p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2))) * 1e3
        return np.asarray(wts * (p.total_variance(k) - w_mkt) + penalty, dtype=float)

    x0 = np.array([max(w_mkt.min() * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1])
    lb = [-1.0, 0.0, -0.999, -2.0, 1e-4]
    ub = [max(1.0, 2 * w_mkt.max()), 5.0, 0.999, 2.0, 5.0]
    best = None
    for m0 in (0.0, k[np.argmin(w_mkt)]):
        for rho0 in (-0.6, -0.2, 0.2):
            x0[2], x0[3] = rho0, m0
            sol = least_squares(
                resid, x0, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=5000
            )
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    p = unpack(best.x)
    fitted = p.implied_vol(k, T)
    grid = np.linspace(k.min() - 0.1, k.max() + 0.1, 201)
    return SVIFit(
        params=p,
        T=T,
        rmse_vol=float(np.sqrt(np.mean((fitted - iv) ** 2))),
        n_points=len(k),
        butterfly_arbitrage_free=bool(np.all(durrleman_g(p, grid) >= -1e-8)),
    )


def synthetic_chain(
    F: float,
    T: float,
    r: float,
    true: SVIParams,
    n: int = 25,
    width: float = 0.25,
    noise_vol: float = 0.002,
    seed: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Option prices generated from a known SVI smile plus quote noise (for tests/demo).

    Strikes are snapped to 25-point NQ option strikes; OTM puts below F, calls above.
    """
    rng = np.random.default_rng(seed)
    k = np.linspace(-width, width * 0.6, n)
    K = np.round(F * np.exp(k) / 25) * 25
    k = np.log(K / F)
    iv_true = true.implied_vol(k, T)
    iv_noisy = iv_true + rng.normal(0, noise_vol, size=n)
    is_call = K >= F
    px = black76.price(F, K, T, r, iv_noisy, is_call)
    return {
        "strike": K,
        "log_moneyness": k,
        "is_call": is_call.astype(float),
        "price": px,
        "iv_true": iv_true,
    }
