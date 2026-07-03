"""Generate the special-function reference table for the Rust core.

Run from the repository root:

    uv run --with mpmath scripts/generate_special_fn_table.py

Values are computed with mpmath at 400 significant digits and rounded to
f64. The Rust unit tests (`crates/core/tests/special_table.rs`) check the
hand-rolled gammaln/digamma/erf/erfc/ndtr/ndtri ports against
this table. Regenerate only when extending coverage; the committed table
is the contract.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Protocol, cast


class _MpNumber(Protocol):
    """Typed subset of ``mpmath.mpf`` used by this generator."""

    def __float__(self) -> float: ...
    def __mul__(self, other: object) -> _MpNumber: ...
    def __rmul__(self, other: object) -> _MpNumber: ...
    def __sub__(self, other: object) -> _MpNumber: ...
    def __rsub__(self, other: object) -> _MpNumber: ...
    def __neg__(self) -> _MpNumber: ...
    def __pow__(self, other: object) -> _MpNumber: ...
    def __lt__(self, other: object) -> bool: ...


class _MpSettings(Protocol):
    """Mutable mpmath precision settings."""

    dps: int


class _MpmathModule(Protocol):
    """Typed subset of mpmath used by this generator."""

    mp: _MpSettings

    def mpf(self, value: object) -> _MpNumber: ...
    def sqrt(self, value: object) -> _MpNumber: ...
    def log(self, value: _MpNumber) -> _MpNumber: ...
    def loggamma(self, value: _MpNumber) -> _MpNumber: ...
    def digamma(self, value: _MpNumber) -> _MpNumber: ...
    def erf(self, value: _MpNumber) -> _MpNumber: ...
    def erfc(self, value: _MpNumber) -> _MpNumber: ...
    def ncdf(self, value: _MpNumber) -> _MpNumber: ...
    def erfinv(self, value: _MpNumber) -> _MpNumber: ...
    def findroot(
        self,
        f: object,
        x0: _MpNumber,
        tol: _MpNumber,
        maxsteps: int,
    ) -> _MpNumber: ...


mp = cast(_MpmathModule, importlib.import_module("mpmath"))

# 400 digits so that 2*p - 1 keeps full precision even for p ~ 1e-300
# (the ndtri reference inverts the CDF via erfinv(2p - 1)).
mp.mp.dps = 400

OUT = Path(__file__).parent.parent / "crates" / "core" / "tests" / "data"

GAMMALN_POINTS = [
    1e-8,
    1e-3,
    0.1,
    0.25,
    0.4999,
    0.5,
    0.6,
    0.9,
    0.9999,
    1.0,
    1.0001,
    1.4616321449683623,
    1.5,
    2.0,
    2.5,
    3.0,
    3.7,
    5.0,
    6.5,
    8.0,
    10.0,
    12.3,
    25.5,
    51.0,
    100.0,
    256.5,
    1000.0,
    12345.6,
    1e6,
    1e10,
]

DIGAMMA_POINTS = [
    1e-6,
    0.01,
    0.1,
    0.25,
    0.49,
    0.5,
    0.75,
    1.0,
    1.4616321449683623,
    1.5,
    2.0,
    2.5,
    3.0,
    5.0,
    7.7,
    10.0,
    33.3,
    100.0,
    1234.5,
    1e6,
]

ERF_POINTS = [
    0.0,
    1e-10,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.99,
    1.0,
    1.01,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.99,
    8.0,
    9.5,
    12.0,
    20.0,
    26.5,
]

NDTR_POINTS = [
    0.0,
    0.1,
    0.5,
    1.0,
    1.4142135623730951,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    37.0,
]

# Covers the three log_ndtr branches: asymptotic series (x < -26),
# erfc-backed log (-26 <= x <= 8), and log1p right tail (x > 8).
LOG_NDTR_POINTS = [
    -500.0,
    -100.0,
    -37.0,
    -30.0,
    -26.5,
    -26.0,
    -25.5,
    -20.0,
    -14.0,
    -8.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    8.0,
    8.5,
    10.0,
    15.0,
    20.0,
    37.0,
]

# Covers both ndtri_exp branches: the exp+ndtri delegation (log_p >= -690)
# and the asymptotic-plus-Newton left tail below the f64 underflow horizon.
# Points right of log(0.5) are excluded: inverting through exp(log_p) loses
# tail resolution near p = 1 and the sampler only queries the left half.
NDTRI_EXP_POINTS = [
    -1e6,
    -1e4,
    -1000.0,
    -750.0,
    -691.0,
    -690.0,
    -689.0,
    -100.0,
    -20.0,
    -5.0,
    -1.0,
    -0.1,
    -1e-3,
]

NDTRI_POINTS = [
    1e-300,
    1e-100,
    1e-30,
    1e-10,
    1e-5,
    1e-3,
    0.01,
    0.1,
    0.1353352832366127,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.86,
    0.8646647167633873,
    0.9,
    0.975,
    0.999,
    0.9999999,
    1.0 - 1e-10,
    1.0 - 1e-15,
]


def _f64(x: _MpNumber) -> float:
    return float(x)


def _ndtri_exp(log_p: float) -> _MpNumber:
    """The x with log(ncdf(x)) = log_p, via findroot at full precision."""
    target = mp.mpf(log_p)
    x0 = -mp.sqrt(-2 * target) if log_p < -1.0 else mp.mpf("0.5")
    return mp.findroot(
        lambda x: mp.log(mp.ncdf(x)) - target,
        x0,
        tol=mp.mpf(10) ** -50,
        maxsteps=200,
    )


def main() -> None:
    table = {
        "gammaln": [[x, _f64(mp.loggamma(mp.mpf(x)))] for x in GAMMALN_POINTS],
        "digamma": [[x, _f64(mp.digamma(mp.mpf(x)))] for x in DIGAMMA_POINTS],
        "erf": [[s * x, _f64(mp.erf(mp.mpf(s * x)))] for x in ERF_POINTS for s in (1.0, -1.0)],
        "erfc": [[s * x, _f64(mp.erfc(mp.mpf(s * x)))] for x in ERF_POINTS for s in (1.0, -1.0)],
        "ndtr": [[s * x, _f64(mp.ncdf(mp.mpf(s * x)))] for x in NDTR_POINTS for s in (1.0, -1.0)],
        "log_ndtr": [[x, _f64(mp.log(mp.ncdf(mp.mpf(x))))] for x in LOG_NDTR_POINTS],
        # ndtri(p): invert the exact normal CDF at the exact f64 value of p
        # (mp.mpf(float) is exact; a decimal literal would shift the target
        # by ~1e-10 in the steep region near p = 1).
        "ndtri": [[p, _f64(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1))] for p in NDTRI_POINTS],
        "ndtri_exp": [[lp, _f64(_ndtri_exp(lp))] for lp in NDTRI_EXP_POINTS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "special_fn_table.json"
    path.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
