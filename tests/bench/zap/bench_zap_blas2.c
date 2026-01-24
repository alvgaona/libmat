/*
 * ZAP Benchmarks for libmat BLAS Level 2 operations
 * Operations: gemv, gemv_t, ger, syr
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

static void fill_random_vec(Vec* v) {
    size_t n = v->rows * v->cols;
    for (size_t i = 0; i < n; i++) {
        v->data[i] = (mat_elem_t)(rand() % 1000) / 100.0f;
    }
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* A;
    Vec* x;
    Vec* y;
    size_t n;
} blas2_ctx_t;

/* ========================================================================== */
/* GEMV: y = alpha*A*x + beta*y                                               */
/* ========================================================================== */

static void iter_gemv(zap_bencher_t* b, void* param) {
    blas2_ctx_t* ctx = (blas2_ctx_t*)param;
    mat_elem_t alpha = 1.0f;
    mat_elem_t beta = 0.0f;

    /* Throughput: read A (n^2), read x (n), write y (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_gemv(ctx->y, alpha, ctx->A, ctx->x, beta);
        zap_black_box(ctx->y->data);
    });
}

/* ========================================================================== */
/* GEMV_T: y = alpha*A^T*x + beta*y                                           */
/* ========================================================================== */

static void iter_gemv_t(zap_bencher_t* b, void* param) {
    blas2_ctx_t* ctx = (blas2_ctx_t*)param;
    mat_elem_t alpha = 1.0f;
    mat_elem_t beta = 0.0f;

    /* Throughput: read A (n^2), read x (n), write y (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_gemv_t(ctx->y, alpha, ctx->A, ctx->x, beta);
        zap_black_box(ctx->y->data);
    });
}

/* ========================================================================== */
/* GER: A += alpha*x*y^T (rank-1 update)                                      */
/* ========================================================================== */

static void iter_ger(zap_bencher_t* b, void* param) {
    blas2_ctx_t* ctx = (blas2_ctx_t*)param;
    mat_elem_t alpha = 1.0f;

    /* Throughput: read x (n), read y (n), read+write A (2*n^2) */
    zap_bencher_set_throughput_bytes(b, (2 * ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_ger(ctx->A, alpha, ctx->x, ctx->y);
        zap_black_box(ctx->A->data);
    });
}

/* ========================================================================== */
/* SYR: A += alpha*x*x^T (symmetric rank-1 update)                            */
/* ========================================================================== */

static void iter_syr(zap_bencher_t* b, void* param) {
    blas2_ctx_t* ctx = (blas2_ctx_t*)param;
    mat_elem_t alpha = 1.0f;

    /* Throughput: read x (n), read+write A lower triangle (n^2/2) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_syr(ctx->A, alpha, ctx->x, 'L');
        zap_black_box(ctx->A->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_gemv_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("gemv");
    zap_group_tag(group, "blas2");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas2_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.x = mat_vec(n);
        ctx.y = mat_vec(n);
        fill_random_mat(ctx.A);
        fill_random_vec(ctx.x);

        zap_benchmark_id_t id = zap_benchmark_id("gemv", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_gemv);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.x);
        mat_free_mat(ctx.y);
    }

    zap_group_finish(group);
}

static void bench_gemv_t_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("gemv_t");
    zap_group_tag(group, "blas2");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas2_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.x = mat_vec(n);
        ctx.y = mat_vec(n);
        fill_random_mat(ctx.A);
        fill_random_vec(ctx.x);

        zap_benchmark_id_t id = zap_benchmark_id("gemv_t", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_gemv_t);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.x);
        mat_free_mat(ctx.y);
    }

    zap_group_finish(group);
}

static void bench_ger_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("ger");
    zap_group_tag(group, "blas2");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas2_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.x = mat_vec(n);
        ctx.y = mat_vec(n);
        fill_random_mat(ctx.A);
        fill_random_vec(ctx.x);
        fill_random_vec(ctx.y);

        zap_benchmark_id_t id = zap_benchmark_id("ger", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_ger);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.x);
        mat_free_mat(ctx.y);
    }

    zap_group_finish(group);
}

static void bench_syr_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("syr");
    zap_group_tag(group, "blas2");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas2_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.x = mat_vec(n);
        ctx.y = NULL;
        fill_random_mat(ctx.A);
        fill_random_vec(ctx.x);

        zap_benchmark_id_t id = zap_benchmark_id("syr", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_syr);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char** argv) {
    srand(42);
    zap_parse_args(argc, argv);

    bench_gemv_group();
    bench_gemv_t_group();
    bench_ger_group();
    bench_syr_group();

    return zap_finalize();
}
