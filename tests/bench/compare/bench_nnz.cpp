/*
 * bench_nnz.cpp - Compare NNZ: libmat vs Eigen
 *
 * result = count(A != 0) (count non-zero elements)
 *
 * Build:
 *   make bench-compare-nnz
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
    Eigen::Map<EigenMatrix>* eA;
    size_t n;
} nnz_ctx_t;

static void fill_random_sparse(mat_elem_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // ~30% sparsity
        data[i] = (rand() % 10 < 3) ? 0.0f : (Scalar)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// libmat: nnz(A)
void bench_libmat(zap_bencher_t* b, void* param) {
    nnz_ctx_t* ctx = (nnz_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, ctx->n * sizeof(Scalar));
    volatile size_t sink;

    ZAP_ITER(b, {
        sink = mat_nnz(ctx->A);
    });
    (void)sink;
}

// Eigen: (A.array() != 0).count()
void bench_eigen(zap_bencher_t* b, void* param) {
    nnz_ctx_t* ctx = (nnz_ctx_t*)param;
    zap_bencher_set_throughput_bytes(b, ctx->n * sizeof(Scalar));
    volatile Eigen::Index sink;

    ZAP_ITER(b, {
        sink = (ctx->eA->array() != 0).count();
    });
    (void)sink;
}

int main(int argc, char** argv) {
    zap_parse_args(argc, argv);
    srand(42);
    Eigen::setNbThreads(1);

    zap_compare_group_t* g = zap_compare_group("nnz");
    zap_compare_set_baseline(g, 1);  // Eigen as baseline

    size_t sizes[] = {1000, 10000, 100000, 1000000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];

        Mat* A = mat_mat(1, n);

        fill_random_sparse(A->data, n);

        Eigen::Map<EigenMatrix> eA(A->data, 1, n);

        nnz_ctx_t ctx = {
            A,
            &eA,
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
    }

    zap_compare_group_finish(g);
    return zap_finalize();
}
