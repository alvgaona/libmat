/*
 * bench_outer.cpp - Compare OUTER: libmat vs Eigen
 *
 * C = x * y^T (outer product)
 *
 * Build:
 *   make bench-compare-outer
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
    Vec* y;
    Mat* C;
    Eigen::Map<EigenVector>* ex;
    Eigen::Map<EigenVector>* ey;
    EigenMatrix* eC;
    size_t m;
    size_t n;
} outer_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: C = x * y^T
void bench_libmat(zap_bencher_t* b, void* param) {
    outer_ctx_t* ctx = (outer_ctx_t*)param;
    // m*n FLOPs (multiplications)
    zap_bencher_set_throughput_elements(b, ctx->m * ctx->n);

    ZAP_ITER(b, {
        mat_outer(ctx->C, ctx->x, ctx->y);
        zap_black_box(ctx->C->data);
    });
}

// Eigen: C = x * y^T
void bench_eigen(zap_bencher_t* b, void* param) {
    outer_ctx_t* ctx = (outer_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, ctx->m * ctx->n);

    ZAP_ITER(b, {
        ctx->eC->noalias() = *ctx->ex * ctx->ey->transpose();
        Scalar* ptr = ctx->eC->data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("outer");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {100, 256, 512, 1000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Vec* x = mat_vec(n);
        Vec* y = mat_vec(n);
        Mat* C = mat_mat(n, n);

        fill_random(x->data, n);
        fill_random(y->data, n);

        Eigen::Map<EigenVector> ex(x->data, n);
        Eigen::Map<EigenVector> ey(y->data, n);
        EigenMatrix eC(n, n);

        outer_ctx_t ctx = {
            x, y, C,
            &ex, &ey, &eC,
            n, n
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);

        zap_compare_end(cmp);

        mat_free_mat(x);
        mat_free_mat(y);
        mat_free_mat(C);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
