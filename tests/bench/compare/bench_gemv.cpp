/*
 * bench_gemv.cpp - Compare GEMV: libmat vs Eigen vs OpenBLAS
 *
 * y = alpha * A * x + beta * y
 *
 * Build:
 *   make bench-compare-gemv
 */

#include <cstdlib>

__attribute__((constructor))
static void force_single_thread(void) {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("GOTO_NUM_THREADS", "1", 1);
}

#include <Eigen/Dense>
#include <cblas.h>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using EigenVector = Eigen::VectorXd;
using Scalar = double;
#define CBLAS_GEMV cblas_dgemv
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_GEMV cblas_sgemv
#endif

typedef struct {
    Mat* A;
    Vec* x;
    Vec* y;
    Vec* y_blas;
    Eigen::Map<EigenMatrix>* eA;
    Eigen::Map<EigenVector>* ex;
    EigenVector* ey;
    size_t m, n;
} gemv_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: y = alpha * A * x + beta * y
void bench_libmat(zap_bencher_t* b, void* param) {
    gemv_ctx_t* ctx = (gemv_ctx_t*)param;
    // 2*m*n FLOPs (multiply-add for each element)
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_gemv(ctx->y, 1.0f, ctx->A, ctx->x, 0.0f);
        zap_black_box(ctx->y->data);
    });
}

// Eigen: y = A * x
void bench_eigen(zap_bencher_t* b, void* param) {
    gemv_ctx_t* ctx = (gemv_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        ctx->ey->noalias() = (*ctx->eA) * (*ctx->ex);
        Scalar* ptr = ctx->ey->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: y = alpha * A * x + beta * y (column-major)
void bench_openblas(zap_bencher_t* b, void* param) {
    gemv_ctx_t* ctx = (gemv_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    int m = (int)ctx->m;
    int n = (int)ctx->n;
    Scalar alpha = 1.0f;
    Scalar beta = 0.0f;

    // Column-major: A is m x n, lda = m
    ZAP_ITER(b, {
        CBLAS_GEMV(CblasColMajor, CblasNoTrans,
                   m, n, alpha,
                   ctx->A->data, m,
                   ctx->x->data, 1,
                   beta,
                   ctx->y_blas->data, 1);
        zap_black_box(ctx->y_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("gemv");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    // Square matrices
    size_t sizes[] = {64, 128, 256, 512, 1024, 2048};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t dim = sizes[s];

        Mat* A = mat_mat(dim, dim);
        Vec* x = mat_vec(dim);
        Vec* y = mat_vec(dim);
        Vec* y_blas = mat_vec(dim);

        fill_random(A->data, dim * dim);
        fill_random(x->data, dim);

        Eigen::Map<EigenMatrix> eA(A->data, dim, dim);
        Eigen::Map<EigenVector> ex(x->data, dim);
        EigenVector ey(dim);

        gemv_ctx_t ctx = {
            A, x, y, y_blas,
            &eA, &ex, &ey,
            dim, dim
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)dim),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(x);
        mat_free_mat(y);
        mat_free_mat(y_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
