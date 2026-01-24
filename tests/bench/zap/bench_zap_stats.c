/*
 * ZAP Benchmarks for libmat statistics operations
 * Operations: mean, std, argmax, argmin
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
    size_t n;
} stats_ctx_t;

/* ========================================================================== */
/* MEAN: result = mean(A)                                                     */
/* ========================================================================== */

static void iter_mean(zap_bencher_t* b, void* param) {
    stats_ctx_t* ctx = (stats_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_mean(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* STD: result = std(A)                                                       */
/* ========================================================================== */

static void iter_std(zap_bencher_t* b, void* param) {
    stats_ctx_t* ctx = (stats_ctx_t*)param;

    /* Throughput: read A (n^2) - needs two passes */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_std(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* ARGMAX: result = argmax(A)                                                 */
/* ========================================================================== */

static void iter_argmax(zap_bencher_t* b, void* param) {
    stats_ctx_t* ctx = (stats_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    size_t result;
    ZAP_ITER(b, {
        result = mat_argmax(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* ARGMIN: result = argmin(A)                                                 */
/* ========================================================================== */

static void iter_argmin(zap_bencher_t* b, void* param) {
    stats_ctx_t* ctx = (stats_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    size_t result;
    ZAP_ITER(b, {
        result = mat_argmin(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_mean_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("mean");
    zap_group_tag(group, "stats");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        stats_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("mean", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_mean);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_std_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("std");
    zap_group_tag(group, "stats");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        stats_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("std", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_std);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_argmax_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("argmax");
    zap_group_tag(group, "stats");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        stats_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("argmax", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_argmax);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_argmin_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("argmin");
    zap_group_tag(group, "stats");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        stats_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("argmin", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_argmin);

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

    bench_mean_group();
    bench_std_group();
    bench_argmax_group();
    bench_argmin_group();

    return zap_finalize();
}
