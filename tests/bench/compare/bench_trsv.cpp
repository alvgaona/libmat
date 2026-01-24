/*
 * bench_trsv.cpp - Compare TRSV: libmat vs Eigen vs OpenBLAS
 *
 * Solve Lx = b (lower triangular solve)
 *
 * Build:
 *   make bench-compare-trsv
 */

#include <cstdlib>
#include <cstring>

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
#define CBLAS_TRSV cblas_dtrsv
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_TRSV cblas_strsv
#endif

typedef struct {
    Mat* L;           // Lower triangular matrix
    Vec* b;           // RHS vector
    Vec* x;           // Solution for libmat
    Vec* x_blas;      // Solution for BLAS (copy of b, modified in place)
    EigenMatrix* eL;  // Eigen copy
    EigenVector* eb;  // Eigen RHS
    size_t n;
} trsv_ctx_t;

// Create well-conditioned lower triangular matrix (column-major)
static void make_tril_colmajor(mat_elem_t* L, size_t n) {
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < j; i++) {
            L[j * n + i] = 0;  // Upper part = 0
        }
        Scalar col_sum = 0;
        for (size_t i = j + 1; i < n; i++) {
            Scalar val = (Scalar)(rand() % 100) / 100.0f - 0.5f;
            L[j * n + i] = val;
            col_sum += (val > 0 ? val : -val);
        }
        L[j * n + j] = col_sum + 1.0f;  // Diagonally dominant
    }
}

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: solve Lx = b
void bench_libmat(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;
    // n^2 FLOPs for triangular solve
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_solve_tril(ctx->x, ctx->L, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

// Eigen: triangularView solve
void bench_eigen(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenVector x = ctx->eL->triangularView<Eigen::Lower>().solve(*ctx->eb);
        Scalar* ptr = x.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: trsv (in-place, overwrites x with solution)
void bench_openblas(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;
    size_t flops = ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    int n = (int)ctx->n;

    ZAP_ITER(b, {
        // Copy b to x_blas (TRSV is in-place)
        memcpy(ctx->x_blas->data, ctx->b->data, ctx->n * sizeof(Scalar));
        // Column-major, lower triangular, no transpose, non-unit diagonal
        CBLAS_TRSV(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                   n, ctx->L->data, n, ctx->x_blas->data, 1);
        zap_black_box(ctx->x_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("trsv");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* L = mat_mat(n, n);
        Vec* b_vec = mat_vec(n);
        Vec* x = mat_vec(n);
        Vec* x_blas = mat_vec(n);

        make_tril_colmajor(L->data, n);
        fill_random(b_vec->data, n);

        // Create Eigen copies
        EigenMatrix eL = Eigen::Map<EigenMatrix>(L->data, n, n);
        EigenVector eb = Eigen::Map<EigenVector>(b_vec->data, n);

        trsv_ctx_t ctx = {
            L, b_vec, x, x_blas,
            &eL, &eb,
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

        mat_free_mat(L);
        mat_free_mat(b_vec);
        mat_free_mat(x);
        mat_free_mat(x_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
