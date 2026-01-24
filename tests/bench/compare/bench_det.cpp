/*
 * bench_det.cpp - Compare DET: libmat vs Eigen
 *
 * result = det(A) (determinant via LU decomposition)
 *
 * Build:
 *   make bench-compare-det
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
} det_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: det(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    det_ctx_t* ctx = (det_ctx_t*)param;
    // LU decomposition: ~2/3 n^3 FLOPs
    size_t flops = 2 * ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = mat_det(ctx->A);
    });
    (void)sink;
}

// Eigen: A.determinant()
void bench_eigen(zap_bencher_t* b, void* param) {
    det_ctx_t* ctx = (det_ctx_t*)param;
    size_t flops = 2 * ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = ctx->eA->determinant();
    });
    (void)sink;
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("det");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);

        fill_random(A->data, n * n);

        Eigen::Map<EigenMatrix> eA(A->data, n, n);

        det_ctx_t ctx = {
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
