/*
 * bench_bilinear.cpp - Compare BILINEAR: libmat vs Eigen
 *
 * result = x^T * A * y (bilinear form)
 *
 * Build:
 *   make bench-compare-bilinear
 */

#include <cstdlib>

#include <Eigen/Dense>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenVector = Eigen::VectorXd;
using EigenMatrix = Eigen::MatrixXd;
using Scalar = double;
#else
using EigenVector = Eigen::VectorXf;
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#endif

typedef struct {
    Vec* x;
    Mat* A;
    Vec* y;
    Eigen::Map<EigenVector>* ex;
    Eigen::Map<EigenMatrix>* eA;
    Eigen::Map<EigenVector>* ey;
    size_t n;
} bilinear_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: x^T * A * y
void bench_libmat(zap_bencher_t* b, void* param) {
    bilinear_ctx_t* ctx = (bilinear_ctx_t*)param;
    // gemv: n^2, dot: n -> ~n^2 FLOPs
    zap_bencher_set_throughput_elements(b, ctx->n * ctx->n);
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = mat_bilinear(ctx->x, ctx->A, ctx->y);
    });
    (void)sink;
}

// Eigen: x.dot(A * y)
void bench_eigen(zap_bencher_t* b, void* param) {
    bilinear_ctx_t* ctx = (bilinear_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, ctx->n * ctx->n);
    volatile Scalar sink;

    ZAP_ITER(b, {
        sink = ctx->ex->dot(*ctx->eA * *ctx->ey);
    });
    (void)sink;
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("bilinear");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Vec* x = mat_vec(n);
        Mat* A = mat_mat(n, n);
        Vec* y = mat_vec(n);

        fill_random(x->data, n);
        fill_random(A->data, n * n);
        fill_random(y->data, n);

        Eigen::Map<EigenVector> ex(x->data, n);
        Eigen::Map<EigenMatrix> eA(A->data, n, n);
        Eigen::Map<EigenVector> ey(y->data, n);

        bilinear_ctx_t ctx = {
            x, A, y,
            &ex, &eA, &ey,
            n
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);

        zap_compare_end(cmp);

        mat_free_mat(x);
        mat_free_mat(A);
        mat_free_mat(y);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
