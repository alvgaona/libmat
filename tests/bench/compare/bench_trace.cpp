/*
 * bench_trace.cpp - Compare TRACE: libmat vs Eigen
 *
 * result = trace(A) (sum of diagonal elements)
 *
 * Build:
 *   make bench-compare-trace
 */

#include <cstdlib>

#include <Eigen/Dense>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using Scalar = double;
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#endif

typedef struct {
    Mat* A;
    Eigen::Map<EigenMatrix>* eA;
    size_t n;
} trace_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: trace(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    trace_ctx_t* ctx = (trace_ctx_t*)param;
    // Only reads n diagonal elements
    zap_bencher_set_throughput_bytes(b, ctx->n * sizeof(Scalar));
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = mat_trace(ctx->A);
    });
    (void)sink;
}

// Eigen: A.trace()
void bench_eigen(zap_bencher_t* b, void* param) {
    trace_ctx_t* ctx = (trace_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, ctx->n * sizeof(Scalar));
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = ctx->eA->trace();
    });
    (void)sink;
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("trace");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);

        fill_random(A->data, n * n);

        Eigen::Map<EigenMatrix> eA(A->data, n, n);

        trace_ctx_t ctx = {
            A,
            &eA,
            n
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);

        zap_compare_end(cmp);

        mat_free_mat(A);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
