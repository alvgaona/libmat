/*
 * bench_scal.cpp - Compare SCAL: libmat vs Eigen vs OpenBLAS
 *
 * x = alpha * x (scale vector)
 *
 * Build:
 *   make bench-compare-scal
 */

#include <cstdlib>

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
using EigenVector = Eigen::VectorXd;
using Scalar = double;
#define CBLAS_SCAL cblas_dscal
#else
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_SCAL cblas_sscal
#endif

typedef struct {
    Vec* x;
    Vec* x_blas;
    EigenVector* ex;
    size_t n;
} scal_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: x = alpha * x
void bench_libmat(zap_bencher_t* b, void* param) {
    scal_ctx_t* ctx = (scal_ctx_t*)param;
    // Memory bandwidth: read n + write n
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(Scalar));

    Scalar alpha = 2.5f;

    ZAP_ITER(b, {
        mat_scale(ctx->x, alpha);
        zap_black_box(ctx->x->data);
    });
}

// Eigen: x *= alpha
void bench_eigen(zap_bencher_t* b, void* param) {
    scal_ctx_t* ctx = (scal_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(Scalar));

    Scalar alpha = 2.5f;

    ZAP_ITER(b, {
        *ctx->ex *= alpha;
        Scalar* ptr = ctx->ex->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: scal
void bench_openblas(zap_bencher_t* b, void* param) {
    scal_ctx_t* ctx = (scal_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(Scalar));

    int n = (int)ctx->n;
    Scalar alpha = 2.5f;

    ZAP_ITER(b, {
        CBLAS_SCAL(n, alpha, ctx->x_blas->data, 1);
        zap_black_box(ctx->x_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("scal");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Vec* x = mat_vec(n);
        Vec* x_blas = mat_vec(n);

        fill_random(x->data, n);
        fill_random(x_blas->data, n);

        EigenVector ex = Eigen::Map<EigenVector>(x->data, n);

        scal_ctx_t ctx = {
            x, x_blas,
            &ex,
            n
        };

        char size_str[32];
        if (n >= 1000000) {
            snprintf(size_str, sizeof(size_str), "%zuM", n / 1000000);
        } else if (n >= 1000) {
            snprintf(size_str, sizeof(size_str), "%zuK", n / 1000);
        } else {
            snprintf(size_str, sizeof(size_str), "%zu", n);
        }

        zap_compare_ctx_t* cmp = zap_compare_begin(
            g, zap_benchmark_id_str("n", size_str),
            &ctx, sizeof(ctx)
        );

        zap_compare_impl(cmp, "libmat", bench_libmat);
        zap_compare_impl(cmp, "Eigen", bench_eigen);
        zap_compare_impl(cmp, "OpenBLAS", bench_openblas);

        zap_compare_end(cmp);

        mat_free_mat(x);
        mat_free_mat(x_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
