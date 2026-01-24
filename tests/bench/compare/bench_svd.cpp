/*
 * bench_svd.cpp - Compare SVD: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * A = U * S * Vt (Singular Value Decomposition)
 *
 * Build:
 *   make bench-compare-svd
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
#include <Eigen/SVD>
#include <lapacke.h>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using EigenVector = Eigen::VectorXd;
using Scalar = double;
#define LAPACK_GESDD LAPACKE_dgesdd
#else
using EigenMatrix = Eigen::MatrixXf;
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define LAPACK_GESDD LAPACKE_sgesdd
#endif

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_work;      // Working copy for libmat
    Mat* U;           // Output U for libmat
    Vec* S;           // Output S for libmat
    Mat* Vt;          // Output Vt for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* U_lap;    // Output U for LAPACK
    Scalar* S_lap;    // Output S for LAPACK
    Scalar* Vt_lap;   // Output Vt for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t m, n;
} svd_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: A = U * S * Vt
void bench_libmat(zap_bencher_t* b, void* param) {
    svd_ctx_t* ctx = (svd_ctx_t*)param;
    // SVD is O(n^3) but constant is high, ~4n^3 roughly
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, ctx->m * ctx->n * sizeof(Scalar));
        mat_svd(ctx->A_work, ctx->U, ctx->S, ctx->Vt);
        zap_black_box(ctx->S->data);
    });
}

// Eigen: BDCSVD (divide & conquer, fast for larger matrices)
// Typedef to avoid comma in macro argument
typedef Eigen::BDCSVD<EigenMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV> EigenBDCSVD;

void bench_eigen(zap_bencher_t* b, void* param) {
    svd_ctx_t* ctx = (svd_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        EigenBDCSVD svd(*ctx->eA);
        EigenVector s = svd.singularValues();
        Scalar* ptr = s.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: GESDD (divide & conquer)
void bench_openblas(zap_bencher_t* b, void* param) {
    svd_ctx_t* ctx = (svd_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int m = (lapack_int)ctx->m;
    lapack_int nn = (lapack_int)ctx->n;

    ZAP_ITER(b, {
        // Copy A to working buffer (GESDD is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, ctx->m * ctx->n * sizeof(Scalar));
        // Column-major, compute full U and Vt
        LAPACK_GESDD(LAPACK_COL_MAJOR, 'A', m, nn, ctx->A_lap->data, m,
                     ctx->S_lap, ctx->U_lap, m, ctx->Vt_lap, nn);
        zap_black_box(ctx->S_lap);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("svd");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    // SVD is expensive, use smaller sizes
    size_t sizes[] = {32, 64, 128, 200};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t m = n;  // Square matrices
        size_t k = m < n ? m : n;

        Mat* A = mat_mat(m, n);
        Mat* A_work = mat_mat(m, n);
        Mat* U = mat_mat(m, m);
        Vec* S = mat_vec(k);
        Mat* Vt = mat_mat(n, n);
        Mat* A_lap = mat_mat(m, n);
        Scalar* U_lap = (Scalar*)malloc(m * m * sizeof(Scalar));
        Scalar* S_lap = (Scalar*)malloc(k * sizeof(Scalar));
        Scalar* Vt_lap = (Scalar*)malloc(n * n * sizeof(Scalar));

        fill_random(A->data, m * n);

        // Create Eigen copy
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, m, n);

        svd_ctx_t ctx = {
            A, A_work, U, S, Vt, A_lap, U_lap, S_lap, Vt_lap,
            &eA,
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
        mat_free_mat(A_work);
        mat_free_mat(U);
        mat_free_mat(S);
        mat_free_mat(Vt);
        mat_free_mat(A_lap);
        free(U_lap);
        free(S_lap);
        free(Vt_lap);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
