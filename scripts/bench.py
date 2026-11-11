"""Benchmark gkpbp against the `ldpc` package on identical shots and priors (min-sum, scaling 0.625, 100 iterations, OSD-CS order 7).

    python scripts/bench.py [shots=500] [threads=1,4,8] [out=bench_results.csv]

Codes: bivariate-bicycle [[144,12,12]] and [[288,12,18]]. Priors are per-shot (uniform in [p_lo, p_hi]), which `ldpc` must receive through
`update_channel_probs` inside a Python loop; gkpbp takes the whole batch in one call. Per-shot times therefore include that Python
overhead for ldpc, which is the realistic cost of per-shot soft information. Failure agreement is checked on the corrections themselves
(fraction of shots with an identical correction), and every gkpbp output is checked to reproduce its syndrome.
Timings vary between machines and runs: report the hardware with any number you quote."""
import sys, time, csv, os, platform, pathlib
import numpy as np
from ldpc import BpOsdDecoder
from gkpbp import BpOsd, has_openmp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from helpers import bb_checks, sample_shots

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 500
threads = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 8]
out = sys.argv[3] if len(sys.argv) > 3 else "bench_results.csv"
CODES = {"[[144,12,12]]": dict(l=12, m=6, a=((3, 0), (0, 1), (0, 2)), b=((0, 3), (1, 0), (2, 0))),
         "[[288,12,18]]": dict(l=12, m=12, a=((3, 0), (0, 2), (0, 7)), b=((0, 3), (1, 0), (2, 0)))}
POINTS = (("low noise", 0.02, 0.05), ("mid noise", 0.05, 0.10), ("high noise", 0.08, 0.14))
print(f"cpu={platform.processor() or platform.machine()} cores={os.cpu_count()} openmp={has_openmp()}")
rows = []
hdr = f"{'code':>14} {'regime':>10} {'ldpc ms':>9} " + " ".join(f"{'gkpbp x'+str(t):>11} {'speedup':>8}" for t in threads) + f" {'identical':>9}"
print(hdr)
for name, cfg in CODES.items():
    HX, HZ = bb_checks(**cfg); N = HZ.shape[1]
    dec = BpOsd(HZ, 100, 0.625, 7)
    for regime, lo, hi in POINTS:
        E, S, P = sample_shots(HZ, shots, lo, hi, seed=11)
        ld = BpOsdDecoder(HZ, error_rate=0.05, max_iter=100, bp_method="minimum_sum", ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=7)
        R = np.zeros((shots, N), np.uint8); t0 = time.perf_counter()
        for b in range(shots): ld.update_channel_probs(P[b]); R[b] = ld.decode(S[b])
        t_ld = (time.perf_counter() - t0) / shots
        ts = []
        for T in threads:
            best = 1e9
            for _ in range(3):
                t0 = time.perf_counter(); C = dec.decode_batch(S, P, threads=T)[0]; best = min(best, (time.perf_counter() - t0) / shots)
            ts.append(best)
        assert ((C @ HZ.T) % 2 == S).all(), "gkpbp output must reproduce the syndrome"
        same = (C == R).all(axis=1).mean()
        print(f"{name:>14} {regime:>10} {t_ld*1e3:9.3f} " + " ".join(f"{t*1e3:11.3f} {t_ld/t:7.1f}x" for t in ts) + f" {same:9.3f}")
        rows.append([name, regime, t_ld * 1e3] + sum(([t * 1e3, t_ld / t] for t in ts), []) + [same])
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["code", "regime", "ldpc_ms"] + sum(([f"gkpbp_x{t}_ms", f"speedup_x{t}"] for t in threads), []) + ["identical_fraction"]); w.writerows(rows)
print("wrote", out)
