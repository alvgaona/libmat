/*
 * ZAP Benchmarks for libmat miscellaneous operations
 * Operations: outer, trace, rank
 */

#define ZAP_IMPLEMENTATION
#include "zap.h"

#define MAT_IMPLEMENTATION
#include "../../../mat.h"

#include <stdlib.h>

/* ========================================================================== */
/* Helper functions                                                           */
/* ========================================================================== */

static void fill_random_vec(Vec* v) {
    size_t n = v->rows * v->cols;
    for (size_t i = 0; i < n; i++) {
        v->data[i] = (mat_elem_t)(rand() % 1000) / 100.0f;
    }
}

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
    Vec* v1;
    Vec* v2;
    Mat* out;
    size_t n;
} outer_ctx_t;

typedef struct {
    Mat* A;
    size_t n;
} mat_ctx_t;

/* ========================================================================== */
/* OUTER: out = v1 * v2^T                                                     */
/* ========================================================================== */

static void iter_outer(zap_bencher_t* b, void* param) {
    outer_ctx_t* ctx = (outer_ctx_t*)param;

    /* Throughput: read v1 (n), read v2 (n), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_outer(ctx->out, ctx->v1, ctx->v2);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* TRACE: result = trace(A)                                                   */
/* ========================================================================== */

static void iter_trace(zap_bencher_t* b, void* param) {
    mat_ctx_t* ctx = (mat_ctx_t*)param;

    /* Throughput: read diagonal (n) */
    zap_bencher_set_throughput_bytes(b, ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_trace(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* RANK: result = rank(A)                                                     */
/* ========================================================================== */

static void iter_rank(zap_bencher_t* b, void* param) {
    mat_ctx_t* ctx = (mat_ctx_t*)param;

    /* Throughput: read A (n^2) - uses SVD internally */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    size_t result;
    ZAP_ITER(b, {
        result = mat_rank(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_outer_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("outer");
    zap_group_tag(group, "misc");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        outer_ctx_t ctx;
        ctx.n = n;
        ctx.v1 = mat_vec(n);
        ctx.v2 = mat_vec(n);
        ctx.out = mat_zeros(n, n);
        fill_random_vec(ctx.v1);
        fill_random_vec(ctx.v2);

        zap_benchmark_id_t id = zap_benchmark_id("outer", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_outer);

        mat_free_mat(ctx.v1);
        mat_free_mat(ctx.v2);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_trace_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("trace");
    zap_group_tag(group, "misc");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        mat_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("trace", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_trace);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_rank_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("rank");
    zap_group_tag(group, "misc");

    /* rank uses SVD - expensive, use smaller sizes */
    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        mat_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("rank", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_rank);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char** argv) {
    srand(42);
    zap_parse_args(argc, argv);

    bench_outer_group();
    bench_trace_group();
    bench_rank_group();

    return zap_finalize();
}
