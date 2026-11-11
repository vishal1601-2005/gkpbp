// gkpbp: a small, dependency-free normalised min-sum belief-propagation decoder with OSD post-processing,
// written to be bit-compatible with the BP stage of the `ldpc` package (flooding schedule, messages <= 0 count as negative)
// and to accept PER-SHOT prior LLRs (needed for GKP analog/soft information).
//
//   channel LLR:  llr_j = log((1 - p_j) / p_j)      (positive = "no error")
//   decision:     x_j = 1  iff  posterior LLR_j <= 0
//   OSD:          columns ordered by increasing posterior LLR; greedy pivots; OSD-0 solves H_B x_B = s, x_rest = 0.
//                 osd_order = lambda > 0 adds a combination sweep: all weight-1 patterns over every non-pivot column and all
//                 weight-2 pairs among the first lambda non-pivot columns, choosing the lowest channel-LLR-weighted solution.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace gkpbp {

static inline int ctz64(uint64_t x) {   // x != 0
#if defined(_MSC_VER)
    unsigned long i; _BitScanForward64(&i, x); return (int)i;
#else
    return __builtin_ctzll(x);
#endif
}

struct Result {
    bool bp_converged = false;   // BP found a syndrome-consistent word
    bool syndrome_ok = false;    // final output reproduces the syndrome (BP or OSD)
    bool osd_used = false;
    int iterations = 0;
};

// In-place 64x64 bit-matrix transpose (Hacker's Delight 7-3). With rows fed in reversed order and columns read in reversed order
// (see osd_cs) it maps LSB-first row words to LSB-first column words.
static inline void transpose64(uint64_t a[64]) {
    int j, k; uint64_t m, t;
    for (j = 32, m = 0x00000000FFFFFFFFULL; j; j >>= 1, m ^= m << j) {
        for (k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            t = (a[k] ^ (a[k | j] >> j)) & m;
            a[k] ^= t; a[k | j] ^= (t << j);
        }
    }
}

class BpOsd {
public:
    // indptr/indices: CSR of the check matrix H (m x n). osd_order: -1 = BP only, 0 = OSD-0, >0 = OSD-CS(order).
    BpOsd(int m, int n, const int32_t* indptr, const int32_t* indices, int max_iter = 100, double ms_scaling = 0.625, int osd_order = -1)
        : m_(m), n_(n), max_iter_(max_iter), alpha_(ms_scaling), osd_order_(osd_order) {
        row_ptr_.assign(indptr, indptr + m + 1);
        const int E = row_ptr_[m];
        var_of_edge_.assign(indices, indices + E);
        col_ptr_.assign(n + 1, 0);
        for (int e = 0; e < E; ++e) {
            if (var_of_edge_[e] < 0 || var_of_edge_[e] >= n) throw std::invalid_argument("column index out of range");
            col_ptr_[var_of_edge_[e] + 1]++;
        }
        for (int j = 0; j < n; ++j) col_ptr_[j + 1] += col_ptr_[j];
        col_edges_.resize(E);
        std::vector<int> fill(col_ptr_.begin(), col_ptr_.end() - 1);
        for (int i = 0; i < m; ++i)
            for (int e = row_ptr_[i]; e < row_ptr_[i + 1]; ++e) col_edges_[fill[var_of_edge_[e]]++] = e;
        v2c_.assign(E, 0.0); c2v_.assign(E, 0.0); post_.assign(n, 0.0);
        W_ = (n + 1 + 63) / 64;
    }

    int m() const { return m_; }
    int n() const { return n_; }

    Result decode(const uint8_t* syndrome, const double* llr, uint8_t* out) {
        Result res;
        const int E = row_ptr_[m_];
        const double INF = std::numeric_limits<double>::infinity();
        for (int e = 0; e < E; ++e) v2c_[e] = llr[var_of_edge_[e]];
        std::fill(c2v_.begin(), c2v_.end(), 0.0);
        for (int j = 0; j < n_; ++j) out[j] = 0;
        for (int it = 1; it <= max_iter_; ++it) {
            for (int i = 0; i < m_; ++i) {                       // check -> variable (normalised min-sum)
                const int b = row_ptr_[i], f = row_ptr_[i + 1];
                int neg = syndrome[i] & 1;
                double min1 = INF, min2 = INF; int arg = -1;
                for (int e = b; e < f; ++e) {
                    const double v = v2c_[e];
                    if (v <= 0) neg ^= 1;
                    const double a = std::fabs(v);
                    if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) { min2 = a; }
                }
                for (int e = b; e < f; ++e) {
                    const int s = neg ^ (v2c_[e] <= 0 ? 1 : 0);
                    const double mag = (e == arg) ? min2 : min1;
                    c2v_[e] = (s ? -1.0 : 1.0) * alpha_ * mag;
                }
            }
            for (int j = 0; j < n_; ++j) {                        // variable update + hard decision
                double t = llr[j];
                for (int k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k) t += c2v_[col_edges_[k]];
                post_[j] = t; out[j] = (t <= 0) ? 1 : 0;
                for (int k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k) { const int e = col_edges_[k]; v2c_[e] = t - c2v_[e]; }
            }
            res.iterations = it;
            if (syndrome_matches(out, syndrome)) { res.bp_converged = true; res.syndrome_ok = true; return res; }
        }
        if (osd_order_ >= 0) osd(syndrome, llr, out, res);
        return res;
    }

private:
    bool syndrome_matches(const uint8_t* x, const uint8_t* s) const {
        for (int i = 0; i < m_; ++i) {
            int par = 0;
            for (int e = row_ptr_[i]; e < row_ptr_[i + 1]; ++e) par ^= x[var_of_edge_[e]];
            if ((par & 1) != (s[i] & 1)) return false;
        }
        return true;
    }

    inline uint64_t* row(int r) { return &M_[(size_t)r * W_]; }

    void osd(const uint8_t* syndrome, const double* llr, uint8_t* out, Result& res) {
        res.osd_used = true;
        order_.resize(n_); std::iota(order_.begin(), order_.end(), 0);
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return post_[a] < post_[b]; });
        pos_.resize(n_);
        for (int p = 0; p < n_; ++p) pos_[order_[p]] = p;
        M_.assign((size_t)m_ * W_, 0ULL);
        for (int i = 0; i < m_; ++i) {
            uint64_t* r = row(i);
            for (int e = row_ptr_[i]; e < row_ptr_[i + 1]; ++e) { const int p = pos_[var_of_edge_[e]]; r[p >> 6] |= 1ULL << (p & 63); }
            if (syndrome[i] & 1) r[n_ >> 6] |= 1ULL << (n_ & 63);
        }
        pivot_pos_.clear(); int rank = 0;
        for (int p = 0; p < n_ && rank < m_; ++p) {
            const int w = p >> 6; const uint64_t bit = 1ULL << (p & 63);
            int piv = -1;
            for (int r = rank; r < m_; ++r) if (M_[(size_t)r * W_ + w] & bit) { piv = r; break; }
            if (piv < 0) continue;
            if (piv != rank) std::swap_ranges(row(piv), row(piv) + W_, row(rank));
            const uint64_t* pr = row(rank);
            for (int r = 0; r < m_; ++r) {
                if (r == rank) continue;
                uint64_t* rr = row(r);
                if (rr[w] & bit) for (int k = w; k < W_; ++k) rr[k] ^= pr[k];   // pivot row is zero left of column p
            }
            pivot_pos_.push_back(p); ++rank;
        }
        const uint64_t augbit = 1ULL << (n_ & 63); const int augw = n_ >> 6;
        bool ok = true;
        for (int r = rank; r < m_; ++r) if (row(r)[augw] & augbit) { ok = false; break; }
        for (int j = 0; j < n_; ++j) out[j] = 0;
        for (int i = 0; i < rank; ++i) if (row(i)[augw] & augbit) out[order_[pivot_pos_[i]]] = 1;
        res.syndrome_ok = ok;
        if (ok && osd_order_ > 0) osd_cs(llr, out, rank);
    }

    void osd_cs(const double* llr, uint8_t* out, int rank) {
        std::vector<char> is_piv(n_, 0);
        for (int p : pivot_pos_) is_piv[p] = 1;
        std::vector<int> np; np.reserve(n_ - rank);
        for (int p = 0; p < n_; ++p) if (!is_piv[p]) np.push_back(p);
        const int NP = (int)np.size(); if (NP == 0) return;
        const int L = std::min(osd_order_, NP);
        std::vector<double> pc(rank); std::vector<char> base(rank);
        for (int i = 0; i < rank; ++i) { const int v = order_[pivot_pos_[i]]; pc[i] = llr[v]; base[i] = out[v]; }
        const int CW = (rank + 63) / 64;
        // Column bit-vectors of the reduced matrix (over the pivot rows) for EVERY column position, built with a blocked 64x64 bit
        // transpose: sequential row reads and sequential writes, instead of one strided bit test per (row, column) pair.
        colT_.resize((size_t)W_ * 64 * CW);
        uint64_t blk[64];
        for (int ib = 0; ib < CW; ++ib) {
            for (int w = 0; w < W_; ++w) {
                for (int r = 0; r < 64; ++r) { const int i = ib * 64 + r; blk[63 - r] = (i < rank) ? row(i)[w] : 0ULL; }
                transpose64(blk);
                for (int c = 0; c < 64; ++c) colT_[(size_t)(w * 64 + c) * CW + ib] = blk[63 - c];
            }
        }
        auto colp = [&](int t) { return &colT_[(size_t)np[t] * CW]; };
        auto delta = [&](const uint64_t* flip) {
            double d = 0;
            for (int k = 0; k < CW; ++k) { uint64_t x = flip[k]; while (x) { const int b = ctz64(x); x &= x - 1; const int i = k * 64 + b; d += base[i] ? -pc[i] : pc[i]; } }
            return d;
        };
        double best = 0.0; int b1 = -1, b2 = -1;
        std::vector<uint64_t> tmp(CW);
        for (int t1 = 0; t1 < NP; ++t1) {
            const uint64_t* c1 = colp(t1);
            const double own1 = llr[order_[np[t1]]];
            const double d = own1 + delta(c1);
            if (d < best - 1e-12) { best = d; b1 = t1; b2 = -1; }
            if (t1 < L) for (int t2 = t1 + 1; t2 < L; ++t2) {
                const uint64_t* c2 = colp(t2);
                for (int k = 0; k < CW; ++k) tmp[k] = c1[k] ^ c2[k];
                const double d2 = own1 + llr[order_[np[t2]]] + delta(tmp.data());
                if (d2 < best - 1e-12) { best = d2; b1 = t1; b2 = t2; }
            }
        }
        if (b1 < 0) return;
        auto apply = [&](int t) {
            out[order_[np[t]]] ^= 1;
            const uint64_t* ct = colp(t);
            for (int i = 0; i < rank; ++i) if (ct[i >> 6] >> (i & 63) & 1ULL) out[order_[pivot_pos_[i]]] ^= 1;
        };
        apply(b1); if (b2 >= 0) apply(b2);
    }

    int m_, n_, max_iter_; double alpha_; int osd_order_; int W_ = 0;
    std::vector<int> row_ptr_, var_of_edge_, col_ptr_, col_edges_;
    std::vector<double> v2c_, c2v_, post_;
    std::vector<int> order_, pos_, pivot_pos_;
    std::vector<uint64_t> M_, colT_;
};

}  // namespace gkpbp
