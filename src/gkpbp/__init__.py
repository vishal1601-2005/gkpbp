"""gkpbp: min-sum BP + OSD decoder with per-shot soft priors (C++17 core, OpenMP batch decoding).

    from gkpbp import BpOsd
    dec = BpOsd(H, max_iter=100, ms_scaling=0.625, osd_order=0)     # -1: BP only, 0: OSD-0, k>0: OSD-CS(k)
    corr, converged, syndrome_ok, iters = dec.decode_batch(syndromes, priors, threads=4)

`priors` are physical error probabilities, shape (n,) shared by all shots or (B, n) per shot (e.g. GKP analog information).
"""
from .decoder import BpOsd, probs_to_llr, has_openmp

__all__ = ["BpOsd", "probs_to_llr", "has_openmp"]
__version__ = "0.1.0"
