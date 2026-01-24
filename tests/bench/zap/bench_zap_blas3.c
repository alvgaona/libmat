/*
 * ZAP Benchmarks for libmat BLAS Level 3 operations
 * Operations: gemm, syrk
 */

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "../../../mat.h"

#include <stdlib.h>

/* ========================================================================== */
/* Helper functions                                                           */
/* ========================================================================== */

static void fill_random_mat(Mat* m) {
    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] = (mat_elem_t)(rand() % 1000) / 100.0f;
    }
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* A;
    Mat* B;
    Mat* C;
    size_t n;
} blas3_ctx_t;

/* ========================================================================== */
/* GEMM: C = alpha*A*B + beta*C                                               */
/* ========================================================================== */

static void iter_gemm(zap_bencher_t* b, void* param) {
    blas3_ctx_t* ctx = (blas3_ctx_t*)param;
    mat_elem_t alpha = 1.0f;
    mat_elem_t beta = 0.0f;

    /* GEMM throughput: 2*n^3 FLOPs, read A (n^2), read B (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_gemm(ctx->C, alpha, ctx->A, ctx->B, beta);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* SYRK: C = alpha*A*A^T + beta*C                                             */
/* ========================================================================== */

static void iter_syrk(zap_bencher_t* b, void* param) {
    blas3_ctx_t* ctx = (blas3_ctx_t*)param;
    mat_elem_t alpha = 1.0f;
    mat_elem_t beta = 0.0f;

    /* SYRK throughput: read A (n^2), write C (n^2/2 symmetric) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_syrk(ctx->C, ctx->A, alpha, beta, 'L');
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_gemm_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("gemm");
    zap_group_tag(group, "blas3");

    /* O(n^3) operations - use smaller sizes */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas3_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = mat_zeros(n, n);
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);
        fill_random_mat(ctx.B);

        zap_benchmark_id_t id = zap_benchmark_id("gemm", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_gemm);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.B);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_syrk_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("syrk");
    zap_group_tag(group, "blas3");

    /* O(n^3) operations - use smaller sizes */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas3_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = NULL;
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("syrk", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_syrk);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char** argv) {
    srand(42);
    zap_parse_args(argc, argv);

    bench_gemm_group();
    bench_syrk_group();

    return zap_finalize();
}
