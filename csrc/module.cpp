// Python bindings: batch decoding with optional per-shot soft priors and OpenMP threading.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "gkpbp/decoder.hpp"

namespace py = pybind11;
using u8arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using f64arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using i32arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

class FastBpOsd {
public:
    FastBpOsd(int m, int n, i32arr indptr, i32arr indices, int max_iter, double ms_scaling, int osd_order)
        : m_(m), n_(n), max_iter_(max_iter), alpha_(ms_scaling), osd_(osd_order),
          indptr_(indptr.data(), indptr.data() + indptr.size()), indices_(indices.data(), indices.data() + indices.size()) {
        if ((int)indptr_.size() != m + 1) throw std::invalid_argument("indptr must have length m+1");
        gkpbp::BpOsd probe(m_, n_, indptr_.data(), indices_.data(), max_iter_, alpha_, osd_);   // validates the graph
    }

    // syndromes: (B, m) uint8.  llr: shape (n,) shared prior OR (B, n) per-shot prior.
    py::tuple decode_batch(u8arr syndromes, f64arr llr, int threads) {
        if (syndromes.ndim() != 2 || syndromes.shape(1) != m_) throw std::invalid_argument("syndromes must have shape (B, m)");
        const int B = (int)syndromes.shape(0);
        const bool per_shot = llr.ndim() == 2;
        if (per_shot ? (llr.shape(0) != B || llr.shape(1) != n_) : (llr.ndim() != 1 || llr.shape(0) != n_))
            throw std::invalid_argument("llr must have shape (n,) or (B, n)");
        py::array_t<uint8_t> corr({B, n_}), conv(B), ok(B); py::array_t<int32_t> iters(B);
        uint8_t* pc = corr.mutable_data(); uint8_t* pv = conv.mutable_data(); uint8_t* po = ok.mutable_data(); int32_t* pi = iters.mutable_data();
        const uint8_t* ps = syndromes.data(); const double* pl = llr.data();
        {
            py::gil_scoped_release release;
            const int T = std::max(1, threads);
#ifdef _OPENMP
#pragma omp parallel num_threads(T)
#endif
            {
                gkpbp::BpOsd dec(m_, n_, indptr_.data(), indices_.data(), max_iter_, alpha_, osd_);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
                for (int b = 0; b < B; ++b) {
                    gkpbp::Result r = dec.decode(ps + (size_t)b * m_, per_shot ? pl + (size_t)b * n_ : pl, pc + (size_t)b * n_);
                    pv[b] = r.bp_converged; po[b] = r.syndrome_ok; pi[b] = r.iterations;
                }
            }
        }
        return py::make_tuple(corr, conv, ok, iters);
    }

    int m() const { return m_; }
    int n() const { return n_; }

private:
    int m_, n_, max_iter_; double alpha_; int osd_;
    std::vector<int32_t> indptr_, indices_;
};

PYBIND11_MODULE(_core, mod) {
    mod.doc() = "gkpbp: min-sum BP + OSD with per-shot soft priors (C++ core)";
    py::class_<FastBpOsd>(mod, "FastBpOsd")
        .def(py::init<int, int, i32arr, i32arr, int, double, int>(), py::arg("m"), py::arg("n"), py::arg("indptr"), py::arg("indices"),
             py::arg("max_iter") = 100, py::arg("ms_scaling") = 0.625, py::arg("osd_order") = -1)
        .def("decode_batch", &FastBpOsd::decode_batch, py::arg("syndromes"), py::arg("llr"), py::arg("threads") = 1)
        .def_property_readonly("m", &FastBpOsd::m).def_property_readonly("n", &FastBpOsd::n);
#ifdef _OPENMP
    mod.attr("openmp") = true;
#else
    mod.attr("openmp") = false;
#endif
}
