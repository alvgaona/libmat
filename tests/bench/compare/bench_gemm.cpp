/*
 * bench_gemm.cpp - Compare GEMM: libmat vs Eigen vs OpenBLAS
 *
 * Build:
 *   make bench-compare-gemm
 *
 * Or manually:
 *   clang++ -O3 -I. -I/opt/homebrew/include/eigen3 -Ideps/openblas/include \
 *           -o bench_gemm tests/bench/compare/bench_gemm.cpp \
 *           -Ldeps/openblas/lib -lopenblas -lm
 */

#include <cstdlib>

// Force single-threaded OpenBLAS before it initializes
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
#define CBLAS_GEMM cblas_dgemm
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#define CBLAS_GEMM cblas_sgemm
#endif

// Context for GEMM benchmarks
typedef struct {
    Mat* A;
    Mat* B;
    Mat* C;
    Mat* C_blas;
    Eigen::Map<EigenMatrix>* eA;
    Eigen::Map<EigenMatrix>* eB;
    EigenMatrix* eC;
    size_t m, n, k;
} gemm_ctx_t;

static void fill_random(Mat* m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: C = alpha * A * B + beta * C
void bench_libmat(zap_bencher_t* b, void* param) {
    gemm_ctx_t* ctx = (gemm_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_gemm(ctx->C, 1.0f, ctx->A, ctx->B, 0.0f);
        zap_black_box(ctx->C->data);
    });
}

// Eigen: C = A * B
void bench_eigen(zap_bencher_t* b, void* param) {
    gemm_ctx_t* ctx = (gemm_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        ctx->eC->noalias() = (*ctx->eA) * (*ctx->eB);
        Scalar* ptr = ctx->eC->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: C = alpha * A * B + beta * C (column-major)
void bench_openblas(zap_bencher_t* b, void* param) {
    gemm_ctx_t* ctx = (gemm_ctx_t*)param;
    size_t flops = 2 * ctx->m * ctx->n * ctx->k;
    zap_bencher_set_throughput_elements(b, flops);

    int m = (int)ctx->m;
    int n = (int)ctx->n;
    int k = (int)ctx->k;
    Scalar alpha = 1.0f;
    Scalar beta = 0.0f;

    ZAP_ITER(b, {
        CBLAS_GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans,
                   m, n, k, alpha,
                   ctx->A->data, m,
                   ctx->B->data, k,
                   beta,
                   ctx->C_blas->data, m);
        zap_black_box(ctx->C_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("gemm");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t dim = sizes[s];

        Mat* A = mat_mat(dim, dim);
        Mat* B = mat_mat(dim, dim);
        Mat* C = mat_mat(dim, dim);
        Mat* C_blas = mat_mat(dim, dim);

        fill_random(A);
        fill_random(B);

        Eigen::Map<EigenMatrix> eA(A->data, dim, dim);
        Eigen::Map<EigenMatrix> eB(B->data, dim, dim);
        EigenMatrix eC(dim, dim);

        gemm_ctx_t ctx = {
            A, B, C, C_blas,
            &eA, &eB, &eC,
            dim, dim, dim
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
        mat_free_mat(B);
        mat_free_mat(C);
        mat_free_mat(C_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
