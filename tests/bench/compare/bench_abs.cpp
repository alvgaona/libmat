/*
 * bench_abs.cpp - Compare ABS: libmat vs Eigen
 *
 * B = abs(A) (element-wise absolute value)
 *
 * Build:
 *   make bench-compare-abs
 */

#include <cstdlib>

#include <Eigen/Dense>

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
using EigenMatrix = Eigen::MatrixXd;
using Scalar = double;
#else
using EigenMatrix = Eigen::MatrixXf;
using Scalar = float;
#endif

typedef struct {
    Mat* A;
    Mat* B;
    EigenMatrix* eA;
    EigenMatrix* eB;
    size_t n;
} abs_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }
}

// libmat: B = abs(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    abs_ctx_t* ctx = (abs_ctx_t*)param;
    // Memory bandwidth: read n + write n
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        mat_abs(ctx->B, ctx->A);
        zap_black_box(ctx->B->data);
    });
}

// Eigen: B = A.cwiseAbs()
void bench_eigen(zap_bencher_t* b, void* param) {
    abs_ctx_t* ctx = (abs_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        *ctx->eB = ctx->eA->cwiseAbs();
        Scalar* ptr = ctx->eB->data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("abs");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(1, n);
        Mat* B = mat_mat(1, n);

        fill_random(A->data, n);

        EigenMatrix eA = Eigen::Map<EigenMatrix>(A->data, 1, n);
        EigenMatrix eB(1, n);

        abs_ctx_t ctx = {
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
