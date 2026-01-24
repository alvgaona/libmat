/*
 * bench_qr.cpp - Compare QR: libmat vs Eigen vs OpenBLAS/LAPACK
 *
 * A = Q * R (QR factorization via Householder)
 *
 * Build:
 *   make bench-compare-qr
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
#define LAPACK_GEQRF LAPACKE_dgeqrf
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#define LAPACK_GEQRF LAPACKE_sgeqrf
#endif

typedef struct {
    Mat* A;           // Original matrix (preserved)
    Mat* A_work;      // Working copy for libmat
    Mat* Q;           // Output Q for libmat
    Mat* R;           // Output R for libmat
    Mat* A_lap;       // Working copy for LAPACK (in-place)
    Scalar* tau;      // Householder scalars for LAPACK
    EigenMatrix* eA;  // Eigen copy
    size_t m, n;
} qr_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: A = Q * R
void bench_libmat(zap_bencher_t* b, void* param) {
    qr_ctx_t* ctx = (qr_ctx_t*)param;
    // ~4n^3/3 FLOPs for square QR
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        memcpy(ctx->A_work->data, ctx->A->data, ctx->m * ctx->n * sizeof(Scalar));
        mat_qr(ctx->A_work, ctx->Q, ctx->R);
        zap_black_box(ctx->R->data);
    });
}

// Eigen: HouseholderQR
void bench_eigen(zap_bencher_t* b, void* param) {
    qr_ctx_t* ctx = (qr_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    ZAP_ITER(b, {
        Eigen::HouseholderQR<EigenMatrix> qr(*ctx->eA);
        EigenMatrix Q = qr.householderQ();
        EigenMatrix R = qr.matrixQR().triangularView<Eigen::Upper>();
        Scalar* ptr = R.data();
        zap_black_box(ptr);
    });
}

// OpenBLAS/LAPACK: GEQRF (in-place, compact form)
void bench_openblas(zap_bencher_t* b, void* param) {
    qr_ctx_t* ctx = (qr_ctx_t*)param;
    size_t n = ctx->n;
    size_t flops = 4 * n * n * n / 3;
    zap_bencher_set_throughput_elements(b, flops);

    lapack_int m = (lapack_int)ctx->m;
    lapack_int nn = (lapack_int)ctx->n;

    ZAP_ITER(b, {
        // Copy A to working buffer (GEQRF is in-place)
        memcpy(ctx->A_lap->data, ctx->A->data, ctx->m * ctx->n * sizeof(Scalar));
        // Column-major
        LAPACK_GEQRF(LAPACK_COL_MAJOR, m, nn, ctx->A_lap->data, m, ctx->tau);
        zap_black_box(ctx->A_lap->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("qr");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t m = n;  // Square matrices
        size_t k = m < n ? m : n;

        Mat* A = mat_mat(m, n);
        Mat* A_work = mat_mat(m, n);
        Mat* Q = mat_mat(m, m);
        Mat* R = mat_mat(m, n);
        Mat* A_lap = mat_mat(m, n);
        Scalar* tau = (Scalar*)malloc(k * sizeof(Scalar));

        fill_random(A->data, m * n);

        // Create Eigen copy
        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, m, n);

        qr_ctx_t ctx = {
            A, A_work, Q, R, A_lap, tau,
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
        mat_free_mat(Q);
        mat_free_mat(R);
        mat_free_mat(A_lap);
        free(tau);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
