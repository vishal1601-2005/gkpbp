"""Print SHA-256 hashes of the decoder outputs used by tests/test_decoder.py::test_outputs_match_golden_hash.
Run ONLY after the ldpc-equivalence tests pass; the hashes then freeze today's verified behaviour so that any later optimisation that
changes even one correction is caught.   Usage (from the repo root):  python scripts/make_golden.py"""
import hashlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from gkpbp import BpOsd
from helpers import bb_checks, sample_shots
HX, HZ = bb_checks()
E, S, P = sample_shots(HZ, 300, 0.04, 0.10, seed=123)
print("GOLDEN = {" + ", ".join(f"{o}: '{hashlib.sha256(BpOsd(HZ, 100, 0.625, o).decode_batch(S, P)[0].tobytes()).hexdigest()}'" for o in (0, 7)) + "}")
