/*
 * bench_axpy.cpp - Compare AXPY: libmat vs Eigen vs OpenBLAS
 *
 * y = alpha * x + y
 *
 * Build:
 *   make bench-compare-axpy
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
#define CBLAS_AXPY cblas_daxpy
#else
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_AXPY cblas_saxpy
#endif

typedef struct {
    Vec* x;
    Vec* y;
    Vec* y_blas;
    Eigen::Map<EigenVector>* ex;
    EigenVector* ey;
    size_t n;
} axpy_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: y = alpha * x + y
void bench_libmat(zap_bencher_t* b, void* param) {
    axpy_ctx_t* ctx = (axpy_ctx_t*)param;
    // 2*n FLOPs (multiply + add)
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    ZAP_ITER(b, {
        mat_axpy(ctx->y, 2.0f, ctx->x);
        zap_black_box(ctx->y->data);
    });
}

// Eigen: y = alpha * x + y
void bench_eigen(zap_bencher_t* b, void* param) {
    axpy_ctx_t* ctx = (axpy_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    ZAP_ITER(b, {
        *ctx->ey += 2.0f * (*ctx->ex);
        Scalar* ptr = ctx->ey->data();
        zap_black_box(ptr);
    });
}

// OpenBLAS: y = alpha * x + y
void bench_openblas(zap_bencher_t* b, void* param) {
    axpy_ctx_t* ctx = (axpy_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    Scalar alpha = 2.0f;
    int n = (int)ctx->n;

    ZAP_ITER(b, {
        CBLAS_AXPY(n, alpha, ctx->x->data, 1, ctx->y_blas->data, 1);
        zap_black_box(ctx->y_blas->data);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("axpy");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Vec* x = mat_vec(n);
        Vec* y = mat_vec(n);
        Vec* y_blas = mat_vec(n);

        fill_random(x->data, n);
        fill_random(y->data, n);
        fill_random(y_blas->data, n);

        Eigen::Map<EigenVector> ex(x->data, n);
        EigenVector ey = Eigen::Map<EigenVector>(y->data, n);

        axpy_ctx_t ctx = {
            x, y, y_blas,
            &ex, &ey,
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
        mat_free_mat(y);
        mat_free_mat(y_blas);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
