import numpy as np
import scipy.sparse as sp
from . import _core

__all__ = ["BpOsd", "probs_to_llr", "has_openmp"]


def has_openmp():
    """True if the extension was built with OpenMP (multi-threaded batch decoding)."""
    return bool(_core.openmp)


def probs_to_llr(p):
    """Error probabilities -> channel LLRs log((1-p)/p), clipped to avoid infinities."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log((1 - p) / p)


class BpOsd:
    """Normalised min-sum BP (flooding schedule) with optional OSD post-processing, over GF(2).

    H           : (m, n) parity-check matrix (dense array or scipy.sparse), entries 0/1
    max_iter    : BP iterations
    ms_scaling  : min-sum scaling factor
    osd_order   : -1 = BP only, 0 = OSD-0, k > 0 = OSD-CS with sweep order k
    """

    def __init__(self, H, max_iter=100, ms_scaling=0.625, osd_order=-1):
        Hc = sp.csr_matrix(H, dtype=np.int32)
        Hc.data %= 2; Hc.eliminate_zeros(); Hc.sort_indices()
        self.m, self.n = Hc.shape
        self._core = _core.FastBpOsd(self.m, self.n, Hc.indptr.astype(np.int32), Hc.indices.astype(np.int32),
                                     int(max_iter), float(ms_scaling), int(osd_order))

    def decode_batch(self, syndromes, priors, threads=1):
        """Decode B syndromes. Returns (corrections (B, n) uint8, bp_converged (B,), syndrome_ok (B,), iterations (B,))."""
        S = np.ascontiguousarray(np.atleast_2d(syndromes), dtype=np.uint8)
        return self._core.decode_batch(S, np.ascontiguousarray(probs_to_llr(priors)), int(threads))

    def decode(self, syndrome, priors):
        """Decode a single syndrome. Returns the correction (n,) uint8."""
        return self.decode_batch(np.asarray(syndrome, dtype=np.uint8)[None, :], priors)[0][0]
