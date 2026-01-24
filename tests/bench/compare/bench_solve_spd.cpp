/*
 * bench_solve_spd.cpp - Compare SOLVE_SPD: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * Solve Ax = b where A is SPD (Cholesky-based)
 *
 * Build:
 *   make bench-compare-solve-spd
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
#define LAPACK_POSV LAPACKE_dposv
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define LAPACK_POSV LAPACKE_sposv
#endif

typedef struct {
    Mat* A;           // Original SPD matrix (preserved)
    Mat* A_work;      // Working copy for libmat (modified)
    Vec* b;           // Original RHS (preserved)
    Vec* b_work;      // Working copy for libmat
    Vec* x;           // Solution for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* x_lap;    // Solution for LAPACK
    EigenMatrix* eA;  // Eigen copy
    EigenVector* eb;  // Eigen RHS
    size_t n;
} solve_spd_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// Make matrix SPD: A = B * B^T + n*I (column-major)
static void make_spd(mat_elem_t* A, size_t n) {
    mat_elem_t* B = (mat_elem_t*)malloc(n * n * sizeof(mat_elem_t));
    fill_random(B, n * n);

    // A = B * B^T (column-major)
    for (size_t j = 0; j < n; j++) {
        for (size_t i = j; i < n; i++) {
            Scalar sum = 0;
            for (size_t k = 0; k < n; k++) {
                sum += B[k * n + i] * B[k * n + j];
            }
            A[j * n + i] = sum;
            A[i * n + j] = sum;
        }
    }

    // Add n*I for numerical stability
    for (size_t i = 0; i < n; i++) {
        A[i * n + i] += (Scalar)n;
    }

    free(B);
}

// libmat: solve Ax = b (SPD)
void bench_libmat(zap_bencher_t* b, void* param) {
    solve_spd_ctx_t* ctx = (solve_spd_ctx_t*)param;
    // Cholesky ~n^3/3 + forward/back sub 2n^2
    size_t flops = ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, ctx->n * ctx->n * sizeof(Scalar));
        memcpy(ctx->b_work->data, ctx->b->data, ctx->n * sizeof(Scalar));
        mat_solve_spd(ctx->x, ctx->A_work, ctx->b_work);
        zap_black_box(ctx->x->data);
    });
}

// Eigen: LLT solve
void bench_eigen(zap_bencher_t* b, void* param) {
    solve_spd_ctx_t* ctx = (solve_spd_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenVector x = ctx->eA->llt().solve(*ctx->eb);
        Scalar* ptr = x.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: POSV (in-place)
void bench_openblas(zap_bencher_t* b, void* param) {
    solve_spd_ctx_t* ctx = (solve_spd_ctx_t*)param;
    size_t flops = ctx->n * ctx->n * ctx->n / 3 + 2 * ctx->n * ctx->n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int n = (lapack_int)ctx->n;

    ZAP_ITER(b, {
        // Copy A and b to working buffers (POSV is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, ctx->n * ctx->n * sizeof(Scalar));
        memcpy(ctx->x_lap, ctx->b->data, ctx->n * sizeof(Scalar));
        // Column-major, lower triangular, nrhs=1
        LAPACK_POSV(LAPACK_COL_MAJOR, 'L', n, 1, ctx->A_lap->data, n, ctx->x_lap, n);
        zap_black_box(ctx->x_lap);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("solve_spd");
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

        make_spd(A->data, n);
        fill_random(b_vec->data, n);

        // Create Eigen copies
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);
        EigenVector eb = Eigen::Map<EigenVector>(b_vec->data, n);

        solve_spd_ctx_t ctx = {
            A, A_work, b_vec, b_work, x, A_lap, x_lap,
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
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
