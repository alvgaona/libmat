/*
 * bench_syrk.cpp - Compare SYRK: libmat vs Eigen vs OpenBLAS
 *
 * C = alpha * A * A^T + beta * C (symmetric rank-k update)
 *
 * Build:
 *   make bench-compare-syrk
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
using Scalar = double;
#define CBLAS_SYRK cblas_dsyrk
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#define CBLAS_SYRK cblas_ssyrk
#endif

typedef struct {
    Mat* A;
    Mat* C;
    Mat* C_blas;
    Eigen::Map<EigenMatrix>* eA;
    EigenMatrix* eC;
    size_t n, k;
} syrk_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: C = alpha * A * A^T + beta * C
void bench_libmat(zap_bencher_t* b, void* param) {
    syrk_ctx_t* ctx = (syrk_ctx_t*)param;
    // n^2 * k FLOPs (approximately, symmetric)
    size_t flops = ctx->n * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_syrk(ctx->C, ctx->A, 1.0f, 0.0f, 'L');
        zap_black_box(ctx->C->data);
    });
}

// Eigen: C = A * A^T
void bench_eigen(zap_bencher_t* b, void* param) {
    syrk_ctx_t* ctx = (syrk_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        ctx->eC->noalias() = (*ctx->eA) * ctx->eA->transpose();
        Scalar* ptr = ctx->eC->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: C = alpha * A * A^T + beta * C (column-major, lower triangle)
void bench_openblas(zap_bencher_t* b, void* param) {
    syrk_ctx_t* ctx = (syrk_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    int n = (int)ctx->n;
    int k = (int)ctx->k;
    Scalar alpha = 1.0f;
    Scalar beta = 0.0f;

    // Column-major: C = A * A^T, A is n x k, lda = n
    ZAP_ITER(b, {
        CBLAS_SYRK(CblasColMajor, CblasLower, CblasNoTrans,
                   n, k, alpha,
                   ctx->A->data, n,
                   beta,
                   ctx->C_blas->data, n);
        zap_black_box(ctx->C_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("syrk");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    // Square: A is n x n, C is n x n
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t k = n;

        Mat* A = mat_mat(n, k);
        Mat* C = mat_mat(n, n);
        Mat* C_blas = mat_mat(n, n);

        fill_random(A->data, n * k);

        Eigen::Map<EigenMatrix> eA(A->data, n, k);
        EigenMatrix eC(n, n);

        syrk_ctx_t ctx = {
            A, C, C_blas,
            &eA, &eC,
            n, k
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(C);
        mat_free_mat(C_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
