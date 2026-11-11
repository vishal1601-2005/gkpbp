import numpy as np


def bb_checks(l=12, m=6, a=((3, 0), (0, 1), (0, 2)), b=((0, 3), (1, 0), (2, 0))):
    """Bivariate-bicycle code (default: the [[144,12,12]] code). Returns (HX, HZ), each (n/2, n)."""
    def mat(terms):
        M = np.zeros((l * m, l * m), np.uint8)
        for i in range(l):
            for j in range(m):
                for (p, q) in terms:
                    M[((i + p) % l) * m + (j + q) % m, i * m + j] ^= 1
        return M
    A, B = mat(a), mat(b)
    return np.hstack([A, B]), np.hstack([B.T, A.T])


def sample_shots(H, shots, p_lo, p_hi, seed):
    """Per-shot priors drawn uniformly from [p_lo, p_hi] per bit; errors drawn from those priors; syndromes H e."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    P = rng.uniform(p_lo, p_hi, size=(shots, n))
    E = (rng.random((shots, n)) < P).astype(np.uint8)
    S = ((E @ H.T) % 2).astype(np.uint8)
    return E, S, P
