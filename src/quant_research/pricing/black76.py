"""Black-76 pricing for European options on futures (e.g. CME NQ options).

All functions are vectorised over numpy arrays. ``T`` is in years, ``r`` is the
continuously-compounded discount rate, ``sigma`` is annualised implied volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

Float = NDArray[np.float64]


def _d1_d2(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike) -> tuple[Float, Float]:
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol_t**2) / vol_t
    return d1, d1 - vol_t


def price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
) -> Float:
    F, K, T, r = (np.asarray(x, dtype=float) for x in (F, K, T, r))
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = np.exp(-r * T)
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(np.asarray(is_call, dtype=bool), call, put)


@dataclass(frozen=True)
class Greeks:
    delta: Float
    gamma: Float
    vega: Float  # per 1.00 vol (divide by 100 for per vol point)
    theta: Float  # per year (divide by 365 for per calendar day)
    rho: Float


def greeks(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
) -> Greeks:
    F, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, r, sigma))
    call = np.asarray(is_call, dtype=bool)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = np.exp(-r * T)
    pdf = norm.pdf(d1)
    delta = np.where(call, df * norm.cdf(d1), -df * norm.cdf(-d1))
    gamma = df * pdf / (F * sigma * np.sqrt(T))
    vega = df * F * pdf * np.sqrt(T)
    p = price(F, K, T, r, sigma, call)
    theta = -df * F * pdf * sigma / (2 * np.sqrt(T)) + r * p
    rho = -T * p
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(
    target: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Implied volatility by Brent's method. Returns NaN outside no-arbitrage bounds."""
    df = np.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if not (intrinsic - 1e-10 <= target < upper):
        return float("nan")

    def f(s: float) -> float:
        return float(price(F, K, T, r, s, is_call)) - target

    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10, maxiter=200))
