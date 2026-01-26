/*
 * bench_eigvals.cpp - Compare eigenvalues: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * Three separate benchmarks:
 * 1. Symmetric eigenvalues only: mat_eigvals_sym vs Eigen SelfAdjoint vs SYEV
 * 2. Non-symmetric eigenvalues: mat_eigvals vs Eigen general vs GEEV
 * 3. Symmetric eigen (vals+vecs): mat_eigen_sym vs Eigen SelfAdjoint vs SYEV
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
using EigenComplex = Eigen::VectorXcd;
using Scalar = double;
#define LAPACK_SYEV LAPACKE_dsyev
#define LAPACK_GEEV LAPACKE_dgeev
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using EigenComplex = Eigen::VectorXcf;
using Scalar = float;
#define LAPACK_SYEV LAPACKE_ssyev
#define LAPACK_GEEV LAPACKE_sgeev
#endif

/* ========================================================================== */
/* Symmetric eigenvalue benchmarks                                            */
/* ========================================================================== */

typedef struct {
    Mat* A;           // Original symmetric matrix (preserved)
    Vec* eig;         // Output eigenvalues for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* eig_lap;  // Output eigenvalues for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} eigvals_sym_ctx_t;

static void fill_symmetric(Mat* A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            Scalar v = (Scalar)rand() / RAND_MAX;
            mat_set_at(A, i, j, v);
            mat_set_at(A, j, i, v);
        }
    }
}

// libmat: mat_eigvals_sym (tridiagonal reduction + QR iteration)
void bench_libmat_sym(zap_bencher_t* b, void* param) {
    eigvals_sym_ctx_t* ctx = (eigvals_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3;  // Tridiagonalization dominates
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_eigvals_sym(ctx->eig, ctx->A);
        zap_black_box(ctx->eig->data);
    });
}

// Eigen: SelfAdjointEigenSolver (optimized for symmetric matrices)
void bench_eigen_sym(zap_bencher_t* b, void* param) {
    eigvals_sym_ctx_t* ctx = (eigvals_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::SelfAdjointEigenSolver<EigenMatrix> solver(*ctx->eA, Eigen::EigenvaluesOnly);
        EigenVector eig = solver.eigenvalues();
        Scalar* ptr = eig.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: SYEV (symmetric eigenvalue solver)
void bench_openblas_sym(zap_bencher_t* b, void* param) {
    eigvals_sym_ctx_t* ctx = (eigvals_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3;
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

/* ========================================================================== */
/* Non-symmetric eigenvalue benchmarks                                        */
/* ========================================================================== */

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_work;      // Working copy for libmat
    Vec* eig;         // Output eigenvalues for libmat (real parts)
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* eig_real; // Real parts of eigenvalues for LAPACK
    Scalar* eig_imag; // Imaginary parts of eigenvalues for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} eigvals_nonsym_ctx_t;

static void fill_random(Mat* A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            Scalar v = (Scalar)rand() / RAND_MAX;
            mat_set_at(A, i, j, v);
        }
    }
}

// libmat: mat_eigvals (Hessenberg reduction + QR iteration)
void bench_libmat_nonsym(zap_bencher_t* b, void* param) {
    eigvals_nonsym_ctx_t* ctx = (eigvals_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n;  // Hessenberg + QR iterations
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, n * n * sizeof(Scalar));
        mat_eigvals(ctx->eig, ctx->A_work);
        zap_black_box(ctx->eig->data);
    });
}

// Eigen: EigenSolver (general non-symmetric)
void bench_eigen_nonsym(zap_bencher_t* b, void* param) {
    eigvals_nonsym_ctx_t* ctx = (eigvals_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::EigenSolver<EigenMatrix> solver(*ctx->eA, false);  // eigenvalues only
        EigenComplex eig = solver.eigenvalues();
        auto ptr = eig.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: GEEV (general eigenvalue solver)
void bench_openblas_nonsym(zap_bencher_t* b, void* param) {
    eigvals_nonsym_ctx_t* ctx = (eigvals_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int nn = (lapack_int)n;

    ZAP_ITER(b, {
        // Copy A to working buffer (GEEV is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, n * n * sizeof(Scalar));
        // Column-major, no left eigenvectors ('N'), no right eigenvectors ('N')
        LAPACK_GEEV(LAPACK_COL_MAJOR, 'N', 'N', nn, ctx->A_lap->data, nn,
                    ctx->eig_real, ctx->eig_imag, nullptr, nn, nullptr, nn);
        zap_black_box(ctx->eig_real);
    });
}

/* ========================================================================== */
/* Symmetric eigendecomposition benchmarks (eigenvalues + eigenvectors)       */
/* ========================================================================== */

typedef struct {
    Mat* A;           // Original symmetric matrix (preserved)
    Mat* V;           // Output eigenvectors for libmat
    Vec* eig;         // Output eigenvalues for libmat
    Mat* A_lap;       // Working copy for LAPACK (stores eigenvectors on exit)
    Scalar* eig_lap;  // Output eigenvalues for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} eigen_sym_ctx_t;

// libmat: mat_eigen_sym (tridiagonal reduction + QR iteration with eigenvectors)
void bench_libmat_eigen_sym(zap_bencher_t* b, void* param) {
    eigen_sym_ctx_t* ctx = (eigen_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3 + 2 * n * n * n;  // Tridiag + eigenvector accumulation
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_eigen_sym(ctx->V, ctx->eig, ctx->A);
        zap_black_box(ctx->V->data);
        zap_black_box(ctx->eig->data);
    });
}

// Eigen: SelfAdjointEigenSolver with eigenvectors
void bench_eigen_eigen_sym(zap_bencher_t* b, void* param) {
    eigen_sym_ctx_t* ctx = (eigen_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3 + 2 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::SelfAdjointEigenSolver<EigenMatrix> solver(*ctx->eA);  // ComputeEigenvectors is default
        EigenVector eig = solver.eigenvalues();
        EigenMatrix vecs = solver.eigenvectors();
        Scalar* eig_ptr = eig.data();
        Scalar* vecs_ptr = vecs.data();
        zap_black_box(eig_ptr);
        zap_black_box(vecs_ptr);
    });
}

// OpenBLAS/LAPACK: SYEV with eigenvectors ('V')
void bench_openblas_eigen_sym(zap_bencher_t* b, void* param) {
    eigen_sym_ctx_t* ctx = (eigen_sym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = (4 * n * n * n) / 3 + 2 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int nn = (lapack_int)n;

    ZAP_ITER(b, {
        // Copy A to working buffer (SYEV overwrites with eigenvectors)
        memcpy(ctx->A_lap->data, ctx->A->data, n * n * sizeof(Scalar));
        // Column-major, compute eigenvectors ('V'), upper triangle ('U')
        LAPACK_SYEV(LAPACK_COL_MAJOR, 'V', 'U', nn, ctx->A_lap->data, nn, ctx->eig_lap);
        zap_black_box(ctx->A_lap->data);
        zap_black_box(ctx->eig_lap);
    });
}

/* ========================================================================== */
/* Non-symmetric eigendecomposition benchmarks (eigenvalues + eigenvectors)   */
/* ========================================================================== */

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_work;      // Working copy for libmat
    Mat* V;           // Output eigenvectors for libmat
    Vec* eig;         // Output eigenvalues for libmat
    Mat* A_lap;       // Working copy for LAPACK
    Scalar* eig_real; // Real parts of eigenvalues for LAPACK
    Scalar* eig_imag; // Imaginary parts of eigenvalues for LAPACK
    Scalar* vl_lap;   // Left eigenvectors for LAPACK (unused)
    Scalar* vr_lap;   // Right eigenvectors for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} eigen_nonsym_ctx_t;

// libmat: mat_eigen (Hessenberg + QR iteration + back-solve)
void bench_libmat_eigen_nonsym(zap_bencher_t* b, void* param) {
    eigen_nonsym_ctx_t* ctx = (eigen_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n + 2 * n * n * n;  // Hessenberg + QR + back-solve
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_eigen(ctx->V, ctx->eig, ctx->A);
        zap_black_box(ctx->V->data);
        zap_black_box(ctx->eig->data);
    });
}

// Eigen: EigenSolver with eigenvectors
void bench_eigen_eigen_nonsym(zap_bencher_t* b, void* param) {
    eigen_nonsym_ctx_t* ctx = (eigen_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n + 2 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::EigenSolver<EigenMatrix> solver(*ctx->eA);  // Compute eigenvectors
        EigenComplex eig = solver.eigenvalues();
        auto vecs = solver.eigenvectors();
        auto eig_ptr = eig.data();
        auto vecs_ptr = vecs.data();
        zap_black_box(eig_ptr);
        zap_black_box(vecs_ptr);
    });
}

// OpenBLAS/LAPACK: GEEV with eigenvectors
void bench_openblas_eigen_nonsym(zap_bencher_t* b, void* param) {
    eigen_nonsym_ctx_t* ctx = (eigen_nonsym_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 10 * n * n * n + 2 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int nn = (lapack_int)n;

    ZAP_ITER(b, {
        // Copy A to working buffer (GEEV is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, n * n * sizeof(Scalar));
        // Column-major, no left eigenvectors ('N'), compute right eigenvectors ('V')
        LAPACK_GEEV(LAPACK_COL_MAJOR, 'N', 'V', nn, ctx->A_lap->data, nn,
                    ctx->eig_real, ctx->eig_imag, ctx->vl_lap, nn, ctx->vr_lap, nn);
        zap_black_box(ctx->vr_lap);
        zap_black_box(ctx->eig_real);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Symmetric eigenvalue comparison
    zap_compare_group_t* g_sym = zap_compare_group("eigvals_sym");
    zap_compare_set_baseline(g_sym, 1);  // Eigen as baseline

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Vec* eig = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* eig_lap = (Scalar*)malloc(n * sizeof(Scalar));

        fill_symmetric(A, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        eigvals_sym_ctx_t ctx = {A, eig, A_lap, eig_lap, &eA, n};

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g_sym, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat_sym);
        zap_compare_impl(cmp, "Eigen", bench_eigen_sym);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas_sym);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(eig);
        mat_free_mat(A_lap);
        free(eig_lap);
    }

    zap_compare_group_finish(g_sym);

    // Non-symmetric eigenvalue comparison
    zap_compare_group_t* g_nonsym = zap_compare_group("eigvals_nonsym");
    zap_compare_set_baseline(g_nonsym, 1);  // Eigen as baseline

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_work = mat_mat(n, n);
        Vec* eig = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* eig_real = (Scalar*)malloc(n * sizeof(Scalar));
        Scalar* eig_imag = (Scalar*)malloc(n * sizeof(Scalar));

        fill_random(A, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        eigvals_nonsym_ctx_t ctx = {A, A_work, eig, A_lap, eig_real, eig_imag, &eA, n};

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g_nonsym, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat_nonsym);
        zap_compare_impl(cmp, "Eigen", bench_eigen_nonsym);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas_nonsym);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(A_work);
        mat_free_mat(eig);
        mat_free_mat(A_lap);
        free(eig_real);
        free(eig_imag);
    }

    zap_compare_group_finish(g_nonsym);

    // Symmetric eigendecomposition (eigenvalues + eigenvectors) comparison
    zap_compare_group_t* g_eigen_sym = zap_compare_group("eigen_sym");
    zap_compare_set_baseline(g_eigen_sym, 1);  // Eigen as baseline

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* V = mat_mat(n, n);
        Vec* eig = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* eig_lap = (Scalar*)malloc(n * sizeof(Scalar));

        fill_symmetric(A, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        eigen_sym_ctx_t ctx = {A, V, eig, A_lap, eig_lap, &eA, n};

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g_eigen_sym, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat_eigen_sym);
        zap_compare_impl(cmp, "Eigen", bench_eigen_eigen_sym);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas_eigen_sym);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(V);
        mat_free_mat(eig);
        mat_free_mat(A_lap);
        free(eig_lap);
    }

    zap_compare_group_finish(g_eigen_sym);

    // Non-symmetric eigendecomposition (eigenvalues + eigenvectors) comparison
    zap_compare_group_t* g_eigen_nonsym = zap_compare_group("eigen_nonsym");
    zap_compare_set_baseline(g_eigen_nonsym, 1);  // Eigen as baseline

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_work = mat_mat(n, n);
        Mat* V = mat_mat(n, n);
        Vec* eig = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* eig_real = (Scalar*)malloc(n * sizeof(Scalar));
        Scalar* eig_imag = (Scalar*)malloc(n * sizeof(Scalar));
        Scalar* vr_lap = (Scalar*)malloc(n * n * sizeof(Scalar));

        fill_random(A, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        eigen_nonsym_ctx_t ctx = {A, A_work, V, eig, A_lap, eig_real, eig_imag, nullptr, vr_lap, &eA, n};

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g_eigen_nonsym, zap_benchmark_id("n", (int64_t)n),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat_eigen_nonsym);
        zap_compare_impl(cmp, "Eigen", bench_eigen_eigen_nonsym);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas_eigen_nonsym);

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(A_work);
        mat_free_mat(V);
        mat_free_mat(eig);
        mat_free_mat(A_lap);
        free(eig_real);
        free(eig_imag);
        free(vr_lap);
    }

    zap_compare_group_finish(g_eigen_nonsym);

    return zap_finalize();
}
