/*
 * bench_chol.cpp - Compare Cholesky: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * L * L^T = A (Cholesky decomposition of SPD matrix)
 *
 * Build:
 *   make bench-compare-chol
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
#include <lapacke.h>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using Scalar = double;
#define LAPACK_POTRF LAPACKE_dpotrf
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#define LAPACK_POTRF LAPACKE_spotrf
#endif

typedef struct {
    Mat* A;           // Original SPD matrix (preserved)
    Mat* L;           // Output for libmat
    Mat* A_work;      // Working copy for LAPACK (in-place)
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} chol_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// Make matrix symmetric positive definite: A = B * B^T + n*I
static void make_spd(Mat* A) {
    size_t n = A->rows;
    Mat* B = mat_mat(n, n);
    fill_random(B->data, n * n);

    // A = B * B^T (column-major)
    for (size_t j = 0; j < n; j++) {
        for (size_t i = j; i < n; i++) {
            Scalar sum = 0;
            for (size_t k = 0; k < n; k++) {
                sum += B->data[k * n + i] * B->data[k * n + j];
            }
            A->data[j * n + i] = sum;
            A->data[i * n + j] = sum;
        }
    }

    // Add n*I for numerical stability
    for (size_t i = 0; i < n; i++) {
        A->data[i * n + i] += (Scalar)n;
    }

    mat_free_mat(B);
}

// libmat: L = chol(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    chol_ctx_t* ctx = (chol_ctx_t*)param;
    // ~n^3/3 FLOPs for Cholesky
    size_t flops = ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_chol(ctx->A, ctx->L);
        zap_black_box(ctx->L->data);
    });
}

// Eigen: LLT decomposition
void bench_eigen(zap_bencher_t* b, void* param) {
    chol_ctx_t* ctx = (chol_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::LLT<EigenMatrix> llt(*ctx->eA);
        EigenMatrix L = llt.matrixL();
        Scalar* ptr = L.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: POTRF (in-place, overwrites input)
void bench_openblas(zap_bencher_t* b, void* param) {
    chol_ctx_t* ctx = (chol_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    int n = (int)ctx->n;

    ZAP_ITER(b, {
        // Copy A to working buffer (POTRF is in-place)
        memcpy(ctx->A_work->data, ctx->A->data, n * n * sizeof(Scalar));
        // Column-major, lower triangle
        LAPACK_POTRF(LAPACK_COL_MAJOR, 'L', n, ctx->A_work->data, n);
        zap_black_box(ctx->A_work->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("chol");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* L = mat_mat(n, n);
        Mat* A_work = mat_mat(n, n);

        make_spd(A);

        // Create Eigen copy
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        chol_ctx_t ctx = {
            A, L, A_work,
            &eA,
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
        mat_free_mat(L);
        mat_free_mat(A_work);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
