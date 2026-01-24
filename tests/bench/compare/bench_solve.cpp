/*
 * bench_solve.cpp - Compare SOLVE: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * Solve Ax = b (LU with partial pivoting)
 *
 * Build:
 *   make bench-compare-solve
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
using EigenVector = Eigen::VectorXd;
using Scalar = double;
#define LAPACK_GESV LAPACKE_dgesv
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define LAPACK_GESV LAPACKE_sgesv
#endif

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_work;      // Working copy for libmat (modified by solve)
    Vec* b;           // Original RHS (preserved)
    Vec* b_work;      // Working copy for libmat
    Vec* x;           // Solution for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* x_lap;    // Solution for LAPACK (also input b)
    lapack_int* ipiv; // Pivot indices for LAPACK
    EigenMatrix* eA;  // Eigen copy
    EigenVector* eb;  // Eigen RHS
    size_t n;
} solve_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// Make matrix diagonally dominant for better numerical stability
static void make_diagonally_dominant(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        Scalar row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                Scalar val = data[j * n + i];  // Column-major
                row_sum += (val > 0 ? val : -val);
            }
        }
        data[i * n + i] = row_sum + 1.0f;  // Column-major diagonal
    }
}

// libmat: solve Ax = b
void bench_libmat(zap_bencher_t* b, void* param) {
    solve_ctx_t* ctx = (solve_ctx_t*)param;
    // ~2/3 n^3 (LU) + 2n^2 (forward/back sub)
    size_t flops = 2 * ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, ctx->n * ctx->n * sizeof(Scalar));
        memcpy(ctx->b_work->data, ctx->b->data, ctx->n * sizeof(Scalar));
        mat_solve(ctx->x, ctx->A_work, ctx->b_work);
        zap_black_box(ctx->x->data);
    });
}

// Eigen: PartialPivLU solve
void bench_eigen(zap_bencher_t* b, void* param) {
    solve_ctx_t* ctx = (solve_ctx_t*)param;
    size_t flops = 2 * ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenVector x = ctx->eA->partialPivLu().solve(*ctx->eb);
        Scalar* ptr = x.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: GESV (in-place, overwrites A and b)
void bench_openblas(zap_bencher_t* b, void* param) {
    solve_ctx_t* ctx = (solve_ctx_t*)param;
    size_t flops = 2 * ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int n = (lapack_int)ctx->n;

    ZAP_ITER(b, {
        // Copy A and b to working buffers (GESV is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, ctx->n * ctx->n * sizeof(Scalar));
        memcpy(ctx->x_lap, ctx->b->data, ctx->n * sizeof(Scalar));
        // Column-major, nrhs=1
        LAPACK_GESV(LAPACK_COL_MAJOR, n, 1, ctx->A_lap->data, n, ctx->ipiv, ctx->x_lap, n);
        zap_black_box(ctx->x_lap);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("solve");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_work = mat_mat(n, n);
        Vec* b_vec = mat_vec(n);
        Vec* b_work = mat_vec(n);
        Vec* x = mat_vec(n);
        Mat* A_lap = mat_mat(n, n);
        Scalar* x_lap = (Scalar*)malloc(n * sizeof(Scalar));
        lapack_int* ipiv = (lapack_int*)malloc(n * sizeof(lapack_int));

        fill_random(A->data, n * n);
        make_diagonally_dominant(A->data, n);
        fill_random(b_vec->data, n);

        // Create Eigen copies
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);
        EigenVector eb = Eigen::Map<EigenVector>(b_vec->data, n);

        solve_ctx_t ctx = {
            A, A_work, b_vec, b_work, x, A_lap, x_lap, ipiv,
            &eA, &eb,
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
        mat_free_mat(A_work);
        mat_free_mat(b_vec);
        mat_free_mat(b_work);
        mat_free_mat(x);
        mat_free_mat(A_lap);
        free(x_lap);
        free(ipiv);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
