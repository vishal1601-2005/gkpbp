# gkpbp

A fast **min-sum belief-propagation + OSD decoder** for binary linear codes (CSS / qLDPC), written in C++17 with OpenMP batch decoding and a
NumPy/SciPy interface. It does one thing the reference `ldpc` package makes awkward: **per-shot soft priors** (a different error probability
for every bit of every shot, e.g. GKP analog information) decoded in a single batched call.

- BP stage **bit-identical** to the `ldpc` package (corrections and convergence flags on every shot; iteration counts on every shot with a nonzero syndrome, see below)
- OSD-0 and OSD-CS(order) post-processing; OSD column extraction uses a blocked 64x64 bit transpose
- Multi-threaded batch decoding; results do not depend on the thread count
- Pure `pip install`: pybind11 build, falls back to single-threaded if the compiler has no OpenMP

```python
import numpy as np
from gkpbp import BpOsd

dec = BpOsd(H, max_iter=100, ms_scaling=0.625, osd_order=7)       # H: (m, n) 0/1 array or scipy.sparse
corr, converged, syndrome_ok, iters = dec.decode_batch(syndromes, priors, threads=4)
# syndromes: (B, m) uint8      priors: error probabilities, shape (n,) shared or (B, n) per shot
# osd_order: -1 = BP only, 0 = OSD-0, k > 0 = OSD-CS(k)
```

## Install
    pip install .            # from a clone; needs a C++17 compiler
    pip install ".[test]"    # adds pytest and ldpc for the test-suite

## How it is validated
`python -m pytest -q` (13+ tests) compares against `ldpc` on a [[144,12,12]] bivariate-bicycle code with per-shot random priors:
- BP corrections identical on every shot; convergence flags identical; iteration counts identical on every shot with a nonzero syndrome
  (for an all-zero syndrome `ldpc` reports a stale `.iter` from an earlier shot, gkpbp reports 1; the corrections are identical)
- exact-tie convention (priors of exactly 0.5, zero-valued messages count as negative) matches `ldpc`
- OSD-0 matches `ldpc` on > 99% of shots, OSD-CS on > 85% (differences are tie-breaks in the reliability sort), with equal mean channel cost
- every output reproduces its syndrome; thread-count invariance; SHA-256 regression hashes on fixed shots
- the test suite was checked with mutation tests (changing the BP scaling, tie convention, OSD-CS threshold or sweep order makes tests fail)

## Speed
`python scripts/bench.py` compares against `ldpc` (OSD-CS order 7, identical shots and priors; the `ldpc` loop includes the Python call and
`update_channel_probs` per shot, which per-shot priors require). Example on a 2-core cloud VM (indicative only; timings vary by machine and run):

| code | noise | ldpc ms/shot | gkpbp 1 thread | gkpbp 4 threads (2 cores) |
|---|---|---|---|---|
| [[144,12,12]] | low | 0.034 | 2.2x | 4.0x |
| [[144,12,12]] | mid | 0.233 | 2.2x | 4.2x |
| [[288,12,18]] | mid | 0.738 | 2.9x | 5.6x |
| [[288,12,18]] | high | 1.555 | 3.5x | 6.6x |

Larger speedups (up to about 20x single-threaded and more with 8 threads) appear for larger codes and the high-noise OSD-heavy regime; see
the GKP-qLDPC threshold study this decoder comes from (https://github.com/vishal1601-2005/gkp-qldpc-thresholds) for a 12-core table and profiling.
Please report your hardware with any number you quote.

## Limitations
- Binary codes only; flooding-schedule normalised min-sum (no layered/serial schedules, no sum-product)
- OSD-CS order is a sweep over weight-1 and limited weight-2 patterns, as in `ldpc`'s `osd_cs`; no higher-order search
- Windows and macOS builds are exercised only by the CI workflow, not by the author's local tests
- The code parameters in `scripts/bench.py` are from the literature (Bravyi et al.); the tests verify n, k and the CSS condition, not the distance

MIT licence.
