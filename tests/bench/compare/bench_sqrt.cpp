/*
 * bench_sqrt.cpp - Compare SQRT: libmat vs Eigen
 *
 * B = sqrt(A) (element-wise square root)
 *
 * Build:
 *   make bench-compare-sqrt
 */

#include <cstdlib>

#include <Eigen/Dense>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenArray = Eigen::ArrayXd;
using Scalar = double;
#else
using EigenArray = Eigen::ArrayXf;
using Scalar = float;
#endif

typedef struct {
    Mat* A;
    Mat* B;
    Eigen::Map<EigenArray>* eA;
    EigenArray* eB;
    size_t n;
} sqrt_ctx_t;

static void fill_random_positive(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX * 100.0f;  // [0, 100]
    }
}

// libmat: B = sqrt(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    sqrt_ctx_t* ctx = (sqrt_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, ctx->n);

    ZAP_ITER(b, {
        mat_sqrt(ctx->B, ctx->A);
        zap_black_box(ctx->B->data);
    });
}

// Eigen: B = A.sqrt()
void bench_eigen(zap_bencher_t* b, void* param) {
    sqrt_ctx_t* ctx = (sqrt_ctx_t*)param;
    zap_bencher_set_throughput_elements(b, ctx->n);

    ZAP_ITER(b, {
        *ctx->eB = ctx->eA->sqrt();
        Scalar* ptr = ctx->eB->data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("sqrt");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(1, n);
        Mat* B = mat_mat(1, n);

        fill_random_positive(A->data, n);

        Eigen::Map<EigenArray> eA(A->data, n);
        EigenArray eB(n);

        sqrt_ctx_t ctx = {
            A, B,
            &eA, &eB,
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

        zap_compare_end(cmp);

        mat_free_mat(A);
        mat_free_mat(B);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
