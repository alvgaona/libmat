/*
 * bench_eigvals.cpp - Compare eigenvalues: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * Compute eigenvalues of symmetric matrices
 *
 * Build:
 *   make bench-compare-eigvals
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
#include <Eigen/Eigenvalues>
#include <lapacke.h>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using EigenVector = Eigen::VectorXd;
using Scalar = double;
#define LAPACK_SYEV LAPACKE_dsyev
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define LAPACK_SYEV LAPACKE_ssyev
#endif

typedef struct {
    Mat* A;           // Original symmetric matrix (preserved)
    Mat* A_work;      // Working copy for libmat
    Vec* eig;         // Output eigenvalues for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* eig_lap;  // Output eigenvalues for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} eigvals_ctx_t;

static void fill_symmetric(Mat* A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            Scalar v = (Scalar)rand() / RAND_MAX;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }
}

// libmat eigenvalues (general Hessenberg QR)
void bench_libmat(zap_bencher_t* b, void* param) {
    eigvals_ctx_t* ctx = (eigvals_ctx_t*)param;
    size_t n = ctx->n;
    // QR algorithm is O(n^3)
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, n * n * sizeof(Scalar));
        mat_eigvals(ctx->eig, ctx->A_work);
        zap_black_box(ctx->eig->data);
    });
}

// libmat eigenvalues (symmetric tridiagonal QR)
void bench_libmat_sym(zap_bencher_t* b, void* param) {
    eigvals_ctx_t* ctx = (eigvals_ctx_t*)param;
    size_t n = ctx->n;
    // Tridiagonal QR is O(n^2) per iteration but uses O(n^3) tridiagonalization
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_eigvals_sym(ctx->eig, ctx->A);
        zap_black_box(ctx->eig->data);
    });
}

// Eigen: SelfAdjointEigenSolver (optimized for symmetric matrices)
void bench_eigen(zap_bencher_t* b, void* param) {
    eigvals_ctx_t* ctx = (eigvals_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::SelfAdjointEigenSolver<EigenMatrix> solver(*ctx->eA, Eigen::EigenvaluesOnly);
        EigenVector eig = solver.eigenvalues();
        Scalar* ptr = eig.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: SYEV (symmetric eigenvalue solver)
void bench_openblas(zap_bencher_t* b, void* param) {
    eigvals_ctx_t* ctx = (eigvals_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int nn = (lapack_int)n;

    ZAP_ITER(b, {
        // Copy A to working buffer (SYEV is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, n * n * sizeof(Scalar));
        // Column-major, compute eigenvalues only ('N'), upper triangle ('U')
        LAPACK_SYEV(LAPACK_COL_MAJOR, 'N', 'U', nn, ctx->A_lap->data, nn, ctx->eig_lap);
        zap_black_box(ctx->eig_lap);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("eigvals");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    // Eigenvalue computation is expensive, use moderate sizes
    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_work = mat_mat(n, n);
        Vec* eig = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* eig_lap = (Scalar*)malloc(n * sizeof(Scalar));

        fill_symmetric(A, n);

        // Create Eigen copy
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        eigvals_ctx_t ctx = {
            A, A_work, eig, A_lap, eig_lap,
            &eA,
            n
        };

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "libmat_sym", bench_libmat_sym);
        zap_compare_impl(cmp, "Eigen", bench_eigen);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(A_work);
        mat_free_mat(eig);
        mat_free_mat(A_lap);
        free(eig_lap);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
