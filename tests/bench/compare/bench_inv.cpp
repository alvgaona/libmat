/*
 * bench_inv.cpp - Compare INV: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * A^(-1) (matrix inverse via LU)
 *
 * Build:
 *   make bench-compare-inv
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
#define LAPACK_GETRF LAPACKE_dgetrf
#define LAPACK_GETRI LAPACKE_dgetri
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#define LAPACK_GETRF LAPACKE_sgetrf
#define LAPACK_GETRI LAPACKE_sgetri
#endif

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_inv;       // Output for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    lapack_int* ipiv; // Pivot indices for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t n;
} inv_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// Make matrix diagonally dominant for invertibility
static void make_invertible(mat_elem_t* data, size_t n) {
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

// libmat: A^(-1)
void bench_libmat(zap_bencher_t* b, void* param) {
    inv_ctx_t* ctx = (inv_ctx_t*)param;
    // ~8/3 n^3 FLOPs for LU + inverse
    size_t flops = 8 * ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        mat_inv(ctx->A_inv, ctx->A);
        zap_black_box(ctx->A_inv->data);
    });
}

// Eigen: inverse()
void bench_eigen(zap_bencher_t* b, void* param) {
    inv_ctx_t* ctx = (inv_ctx_t*)param;
    size_t flops = 8 * ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenMatrix inv = ctx->eA->inverse();
        Scalar* ptr = inv.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: GETRF + GETRI (in-place)
void bench_openblas(zap_bencher_t* b, void* param) {
    inv_ctx_t* ctx = (inv_ctx_t*)param;
    size_t flops = 8 * ctx->n * ctx->n * ctx->n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int n = (lapack_int)ctx->n;

    ZAP_ITER(b, {
        // Copy A to working buffer (in-place operations)
        memcpy(ctx->A_lap->data, ctx->A->data, ctx->n * ctx->n * sizeof(Scalar));
        // Column-major: LU factorization then inversion
        LAPACK_GETRF(LAPACK_COL_MAJOR, n, n, ctx->A_lap->data, n, ctx->ipiv);
        LAPACK_GETRI(LAPACK_COL_MAJOR, n, ctx->A_lap->data, n, ctx->ipiv);
        zap_black_box(ctx->A_lap->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("inv");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(n, n);
        Mat* A_inv = mat_mat(n, n);
        Mat* A_lap = mat_mat(n, n);
        lapack_int* ipiv = (lapack_int*)malloc(n * sizeof(lapack_int));

        fill_random(A->data, n * n);
        make_invertible(A->data, n);

        // Create Eigen copy
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, n, n);

        inv_ctx_t ctx = {
            A, A_inv, A_lap, ipiv,
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
        mat_free_mat(A_inv);
        mat_free_mat(A_lap);
        free(ipiv);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
