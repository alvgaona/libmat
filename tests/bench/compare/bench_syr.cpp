/*
 * bench_syr.cpp - Compare SYR: libmat vs Eigen vs OpenBLAS
 *
 * A += alpha * x * x^T (symmetric rank-1 update)
 *
 * Build:
 *   make bench-compare-syr
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
#define CBLAS_SYR cblas_dsyr
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_SYR cblas_ssyr
#endif

typedef struct {
    Mat* A;
    Mat* A_blas;
    Vec* x;
    EigenMatrix* eA;
    Eigen::Map<EigenVector>* ex;
    size_t n;
} syr_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: A += alpha * x * x^T
void bench_libmat(zap_bencher_t* b, void* param) {
    syr_ctx_t* ctx = (syr_ctx_t*)param;
    // n^2 FLOPs (symmetric)
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    Scalar alpha = 1.0f;

    ZAP_ITER(b, {
        mat_syr(ctx->A, alpha, ctx->x, 'L');
        zap_black_box(ctx->A->data);
    });
}

// Eigen: selfadjointView rankUpdate
void bench_eigen(zap_bencher_t* b, void* param) {
    syr_ctx_t* ctx = (syr_ctx_t*)param;
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    Scalar alpha = 1.0f;

    ZAP_ITER(b, {
        ctx->eA->selfadjointView<Eigen::Lower>().rankUpdate(*ctx->ex, alpha);
        Scalar* ptr = ctx->eA->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: syr
void bench_openblas(zap_bencher_t* b, void* param) {
    syr_ctx_t* ctx = (syr_ctx_t*)param;
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    int n = (int)ctx->n;
    Scalar alpha = 1.0f;

    // Column-major, lower triangular
    ZAP_ITER(b, {
        CBLAS_SYR(CblasColMajor, CblasLower, n, alpha,
                  ctx->x->data, 1, ctx->A_blas->data, n);
        zap_black_box(ctx->A_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("syr");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_blas = mat_mat(n, n);
        Vec* x = mat_vec(n);

        fill_random(x->data, n);
        mat_fill(A, 0);
        mat_fill(A_blas, 0);

        EigenMatrix eA = EigenMatrix::Zero(n, n);
        Eigen::Map<EigenVector> ex(x->data, n);

        syr_ctx_t ctx = {
            A, A_blas, x,
            &eA, &ex,
            n
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
        mat_free_mat(A_blas);
        mat_free_mat(x);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
