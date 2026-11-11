"""Validation against the reference `ldpc` package plus exact-output regression and input checks."""
import hashlib
import numpy as np
import pytest
from ldpc import BpDecoder, BpOsdDecoder
from gkpbp import BpOsd, has_openmp
from helpers import bb_checks, sample_shots

HX, HZ = bb_checks()
M, N = HZ.shape


def _ldpc_bp(S, P, osd=None):
    if osd is None:
        ref = BpDecoder(HZ, error_rate=0.05, max_iter=100, bp_method="minimum_sum", ms_scaling_factor=0.625)
    else:
        ref = BpOsdDecoder(HZ, error_rate=0.05, max_iter=100, bp_method="minimum_sum", ms_scaling_factor=0.625,
                           osd_method="osd_0" if osd == 0 else "osd_cs", osd_order=osd)
    out = np.zeros((len(S), N), np.uint8); conv = np.zeros(len(S), bool); it = np.zeros(len(S), int)
    for b in range(len(S)):
        ref.update_channel_probs(P[b]); out[b] = ref.decode(S[b])
        if osd is None: conv[b] = ref.converge; it[b] = ref.iter
    return out, conv, it


@pytest.mark.parametrize("p_lo,p_hi", [(0.01, 0.03), (0.03, 0.08), (0.02, 0.12)])
def test_bp_stage_is_bit_identical_to_ldpc_with_per_shot_priors(p_lo, p_hi):
    E, S, P = sample_shots(HZ, 300, p_lo, p_hi, seed=1)
    R, conv, it = _ldpc_bp(S, P)
    C, cv, ok, iters = BpOsd(HZ, 100, 0.625, -1).decode_batch(S, P)
    assert (C == R).all()                                      # corrections: bit-identical on every shot
    assert (cv.astype(bool) == conv).all()                     # convergence flags: identical on every shot
    nz = S.any(axis=1)                                         # iteration counts: identical whenever the syndrome is nonzero.
    assert (iters[nz] == it[nz]).all()                         # (For an all-zero syndrome `ldpc` leaves `.iter` stale from the
    assert (iters[~nz] == 1).all()                             #  previous shot; gkpbp reports 1 iteration.)


def test_exact_ties_follow_the_ldpc_convention():
    """Priors of exactly 0.5 give zero LLRs, so zero-valued messages occur: they must count as negative, as in ldpc."""
    E, S, P = sample_shots(HZ, 300, 0.03, 0.10, seed=8)
    rng = np.random.default_rng(9)
    P[rng.random(P.shape) < 0.15] = 0.5
    E = (rng.random(P.shape) < P).astype(np.uint8); S = ((E @ HZ.T) % 2).astype(np.uint8)
    R, conv, it = _ldpc_bp(S, P)
    C, cv, ok, iters = BpOsd(HZ, 100, 0.625, -1).decode_batch(S, P)
    assert (C == R).all() and (cv.astype(bool) == conv).all()


def test_shared_prior_equals_broadcast_per_shot_prior():
    E, S, P = sample_shots(HZ, 100, 0.05, 0.05, seed=2)
    dec = BpOsd(HZ, 100, 0.625, 3)
    a = dec.decode_batch(S, np.full(N, 0.05))[0]; b = dec.decode_batch(S, np.full((100, N), 0.05))[0]
    assert (a == b).all()


def test_osd0_matches_ldpc_and_always_satisfies_syndrome():
    E, S, P = sample_shots(HZ, 400, 0.05, 0.10, seed=3)
    R, _, _ = _ldpc_bp(S, P, osd=0)
    C, cv, ok, it = BpOsd(HZ, 100, 0.625, 0).decode_batch(S, P)
    assert ((C @ HZ.T) % 2 == S).all()                       # output always reproduces the syndrome
    assert ok.all()
    assert (C == R).all(axis=1).mean() > 0.99                # only tie-breaking in the column sort may differ


def test_osd_cs_agrees_with_ldpc_and_is_not_costlier():
    E, S, P = sample_shots(HZ, 300, 0.06, 0.12, seed=4)
    R, _, _ = _ldpc_bp(S, P, osd=7)
    C, cv, ok, it = BpOsd(HZ, 100, 0.625, 7).decode_batch(S, P)
    assert ((C @ HZ.T) % 2 == S).all()
    assert (C == R).all(axis=1).mean() > 0.85
    llr = np.log((1 - P) / P)                                 # channel cost of a correction: sum of LLRs on flipped bits
    cost_c, cost_r = (C * llr).sum(1).mean(), (R * llr).sum(1).mean()
    assert abs(cost_c - cost_r) / cost_r < 0.01


def test_batch_decoding_is_thread_invariant():
    E, S, P = sample_shots(HZ, 200, 0.05, 0.10, seed=5)
    dec = BpOsd(HZ, 100, 0.625, 3)
    a = dec.decode_batch(S, P, threads=1)[0]; b = dec.decode_batch(S, P, threads=3)[0]
    assert (a == b).all()


def test_single_shot_api_matches_batch():
    E, S, P = sample_shots(HZ, 20, 0.05, 0.10, seed=6)
    dec = BpOsd(HZ, 100, 0.625, 0)
    batch = dec.decode_batch(S, P)[0]
    for b in range(20): assert (dec.decode(S[b], P[b]) == batch[b]).all()


def test_input_validation():
    dec = BpOsd(HZ, 100, 0.625, 0)
    with pytest.raises(Exception): dec.decode_batch(np.zeros((3, M + 1), np.uint8), np.full(N, 0.05))
    with pytest.raises(Exception): dec.decode_batch(np.zeros((3, M), np.uint8), np.full(N + 1, 0.05))
    with pytest.raises(Exception): dec.decode_batch(np.zeros((3, M), np.uint8), np.full((4, N), 0.05))


def test_accepts_sparse_and_dense_checks_identically():
    import scipy.sparse as sp
    E, S, P = sample_shots(HZ, 50, 0.05, 0.10, seed=7)
    a = BpOsd(HZ, 100, 0.625, 0).decode_batch(S, P)[0]; b = BpOsd(sp.csr_matrix(HZ), 100, 0.625, 0).decode_batch(S, P)[0]
    assert (a == b).all()


def test_extension_reports_openmp_flag():
    assert isinstance(has_openmp(), bool)


GOLDEN = {0: '0ea752f4b1f7620fa09e7985e31fa33fc512c7c73983b470b94207e92661184c', 7: 'bbe6a5327eeb0609d94f28aee55a7c54decc0188aaa46574859729f32ab28a9d'}   # scripts/make_golden.py, recorded after the ldpc-equivalence tests passed

@pytest.mark.skipif(not GOLDEN, reason="no golden hashes recorded")
@pytest.mark.parametrize("osd", [0, 7])
def test_outputs_match_golden_hash(osd):
    E, S, P = sample_shots(HZ, 300, 0.04, 0.10, seed=123)
    C = BpOsd(HZ, 100, 0.625, osd).decode_batch(S, P)[0]
    assert hashlib.sha256(C.tobytes()).hexdigest() == GOLDEN[osd]


def test_test_codes_are_valid_css_codes_with_expected_dimension():
    """The BB test code: HX HZ^T = 0 and k = n - rank(HX) - rank(HZ) = 12 (distance is taken from the literature, not checked here)."""
    def rank(M):
        M = M.copy() % 2; r = 0
        for c in range(M.shape[1]):
            p = np.nonzero(M[r:, c])[0]
            if len(p) == 0: continue
            M[[r, r + p[0]]] = M[[r + p[0], r]]
            for i in np.nonzero(M[:, c])[0]:
                if i != r: M[i] ^= M[r]
            r += 1
            if r == M.shape[0]: break
        return r
    assert not ((HX @ HZ.T) % 2).any()
    assert HX.shape[1] - rank(HX) - rank(HZ) == 12
