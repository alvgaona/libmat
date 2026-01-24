/*
 * bench_transpose.cpp - Compare TRANSPOSE: libmat vs Eigen
 *
 * B = A^T (matrix transpose)
 *
 * Build:
 *   make bench-compare-transpose
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
    Mat* B;
    Eigen::Map<EigenMatrix>* eA;
    EigenMatrix* eB;
    size_t n;
} transpose_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: B = A^T
void bench_libmat(zap_bencher_t* b, void* param) {
    transpose_ctx_t* ctx = (transpose_ctx_t*)param;
    // Memory bandwidth: read n^2 + write n^2
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        mat_t(ctx->B, ctx->A);
        zap_black_box(ctx->B->data);
    });
}

// Eigen: B = A.transpose()
void bench_eigen(zap_bencher_t* b, void* param) {
    transpose_ctx_t* ctx = (transpose_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        ctx->eB->noalias() = ctx->eA->transpose();
        Scalar* ptr = ctx->eB->data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("transpose");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {64, 128, 256, 512, 1024, 2048};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* B = mat_mat(n, n);

        fill_random(A->data, n * n);

        Eigen::Map<EigenMatrix> eA(A->data, n, n);
        EigenMatrix eB(n, n);

        transpose_ctx_t ctx = {
            A, B,
            &eA, &eB,
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
        mat_free_mat(B);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
