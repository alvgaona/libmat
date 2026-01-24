/*
 * bench_dot.cpp - Compare DOT: libmat vs Eigen vs OpenBLAS
 *
 * result = x . y (dot product)
 *
 * Build:
 *   make bench-compare-dot
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
#define CBLAS_DOT cblas_ddot
#else
using EigenVector = Eigen::VectorXf;
using Scalar = float;
#define CBLAS_DOT cblas_sdot
#endif

typedef struct {
    Vec* x;
    Vec* y;
    Eigen::Map<EigenVector>* ex;
    Eigen::Map<EigenVector>* ey;
    size_t n;
} dot_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: x . y
void bench_libmat(zap_bencher_t* b, void* param) {
    dot_ctx_t* ctx = (dot_ctx_t*)param;
    // 2*n FLOPs (multiply + add)
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    ZAP_ITER(b, {
        mat_elem_t result = mat_dot(ctx->x, ctx->y);
        zap_black_box(result);
    });
}

// Eigen: x . y
void bench_eigen(zap_bencher_t* b, void* param) {
    dot_ctx_t* ctx = (dot_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    ZAP_ITER(b, {
        Scalar result = ctx->ex->dot(*ctx->ey);
        zap_black_box(result);
    });
}

// OpenBLAS: x . y
void bench_openblas(zap_bencher_t* b, void* param) {
    dot_ctx_t* ctx = (dot_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, 2 * ctx->n);

    int n = (int)ctx->n;

    ZAP_ITER(b, {
        Scalar result = CBLAS_DOT(n, ctx->x->data, 1, ctx->y->data, 1);
        zap_black_box(result);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("dot");
    zap_compare_set_baseline(g, 2);  // OpenBLAS as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Vec* x = mat_vec(n);
        Vec* y = mat_vec(n);

        fill_random(x->data, n);
        fill_random(y->data, n);

        Eigen::Map<EigenVector> ex(x->data, n);
        Eigen::Map<EigenVector> ey(y->data, n);

        dot_ctx_t ctx = {
            x, y,
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
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
