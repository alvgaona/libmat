/*
 * bench_add.cpp - Compare ADD: libmat vs Eigen
 *
 * C = A + B (element-wise addition)
 *
 * Build:
 *   make bench-compare-add
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
    Mat* C;
    Eigen::Map<EigenMatrix>* eA;
    Eigen::Map<EigenMatrix>* eB;
    EigenMatrix* eC;
    size_t n;
} add_ctx_t;

static void fill_random(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (Scalar)rand() / RAND_MAX;
    }
}

// libmat: C = A + B
void bench_libmat(zap_bencher_t* b, void* param) {
    add_ctx_t* ctx = (add_ctx_t*)param;
    // Memory bandwidth: read 2n + write n
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        mat_add(ctx->C, ctx->A, ctx->B);
        zap_black_box(ctx->C->data);
    });
}

// Eigen: C = A + B
void bench_eigen(zap_bencher_t* b, void* param) {
    add_ctx_t* ctx = (add_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * sizeof(Scalar));

    ZAP_ITER(b, {
        ctx->eC->noalias() = *ctx->eA + *ctx->eB;
        Scalar* ptr = ctx->eC->data();
        zap_black_box(ptr);
    });
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("add");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(1, n);
        Mat* B = mat_mat(1, n);
        Mat* C = mat_mat(1, n);

        fill_random(A->data, n);
        fill_random(B->data, n);

        Eigen::Map<EigenMatrix> eA(A->data, 1, n);
        Eigen::Map<EigenMatrix> eB(B->data, 1, n);
        EigenMatrix eC(1, n);

        add_ctx_t ctx = {
            A, B, C,
            &eA, &eB, &eC,
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
        mat_free_mat(C);
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
