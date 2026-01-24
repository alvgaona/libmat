/*
 * bench_pinv.cpp - Compare PINV: libmat vs Eigen
 *
 * B = pinv(A) (Moore-Penrose pseudoinverse via SVD)
 *
 * Build:
 *   make bench-compare-pinv
 */

#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/SVD>

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

// Typedef to avoid comma issues in macro
typedef Eigen::JacobiSVD<EigenMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV> EigenJacobiSVD;

typedef struct {
    Mat* A;
    Mat* B;
    EigenMatrix* eA;
    size_t m;
    size_t n;
} pinv_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: B = pinv(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    pinv_ctx_t* ctx = (pinv_ctx_t*)param;
    // SVD complexity: O(mn^2) for m >= n
    size_t k = ctx->m < ctx->n ? ctx->m : ctx->n;
    size_t flops = ctx->m * ctx->n * k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_pinv(ctx->B, ctx->A);
        zap_black_box(ctx->B->data);
    });
}

// Eigen: pinv via JacobiSVD
void bench_eigen(zap_bencher_t* b, void* param) {
    pinv_ctx_t* ctx = (pinv_ctx_t*)param;
    size_t k = ctx->m < ctx->n ? ctx->m : ctx->n;
    size_t flops = ctx->m * ctx->n * k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenJacobiSVD svd(*ctx->eA, Eigen::ComputeFullU | Eigen::ComputeFullV);
        EigenMatrix eB = svd.solve(EigenMatrix::Identity(ctx->m, ctx->m));
        Scalar* ptr = eB.data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("pinv");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {10, 20, 50, 100, 200};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* B = mat_mat(n, n);

        fill_random(A->data, n * n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        pinv_ctx_t ctx = {
            A, B,
            &eA,
            n, n
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
