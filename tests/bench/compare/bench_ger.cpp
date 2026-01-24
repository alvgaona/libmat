/*
 * bench_ger.cpp - Compare GER: libmat vs Eigen vs OpenBLAS
 *
 * A += alpha * x * y^T (rank-1 update / outer product)
 *
 * Build:
 *   make bench-compare-ger
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
#define CBLAS_GER cblas_dger
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_GER cblas_sger
#endif

typedef struct {
    Mat* A;
    Mat* A_blas;
    Vec* x;
    Vec* y;
    EigenMatrix* eA;
    Eigen::Map<EigenVector>* ex;
    Eigen::Map<EigenVector>* ey;
    size_t m, n;
} ger_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: A += alpha * x * y^T
void bench_libmat(zap_bencher_t* b, void* param) {
    ger_ctx_t* ctx = (ger_ctx_t*)param;
    // 2*m*n FLOPs (multiply + add)
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    Scalar alpha = 2.5f;

    ZAP_ITER(b, {
        mat_ger(ctx->A, alpha, ctx->x, ctx->y);
        zap_black_box(ctx->A->data);
    });
}

// Eigen: A += alpha * x * y^T
void bench_eigen(zap_bencher_t* b, void* param) {
    ger_ctx_t* ctx = (ger_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    Scalar alpha = 2.5f;

    ZAP_ITER(b, {
        ctx->eA->noalias() += alpha * (*ctx->ex) * ctx->ey->transpose();
        Scalar* ptr = ctx->eA->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: ger
void bench_openblas(zap_bencher_t* b, void* param) {
    ger_ctx_t* ctx = (ger_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    int m = (int)ctx->m;
    int n = (int)ctx->n;
    Scalar alpha = 2.5f;

    // Column-major: A is m x n, lda = m
    ZAP_ITER(b, {
        CBLAS_GER(CblasColMajor, m, n, alpha,
                  ctx->x->data, 1, ctx->y->data, 1,
                  ctx->A_blas->data, m);
        zap_black_box(ctx->A_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("ger");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t m = n;  // Square

        Mat* A = mat_mat(m, n);
        Mat* A_blas = mat_mat(m, n);
        Vec* x = mat_vec(m);
        Vec* y = mat_vec(n);

        fill_random(A->data, m * n);
        fill_random(A_blas->data, m * n);
        fill_random(x->data, m);
        fill_random(y->data, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, m, n);
        Eigen::Map<EigenVector> ex(x->data, m);
        Eigen::Map<EigenVector> ey(y->data, n);

        ger_ctx_t ctx = {
            A, A_blas, x, y,
            &eA, &ex, &ey,
            m, n
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
        mat_free_mat(y);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
