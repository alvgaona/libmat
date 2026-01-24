/*
 * ZAP Benchmarks for libmat reduction operations
 * Operations: sum, min, max, norm_fro
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
} reduction_ctx_t;

/* ========================================================================== */
/* SUM: result = sum of all elements                                          */
/* ========================================================================== */

static void iter_sum(zap_bencher_t* b, void* param) {
    reduction_ctx_t* ctx = (reduction_ctx_t*)param;

    /* Throughput: read all n^2 elements */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_sum(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* MIN: result = minimum element                                              */
/* ========================================================================== */

static void iter_min(zap_bencher_t* b, void* param) {
    reduction_ctx_t* ctx = (reduction_ctx_t*)param;

    /* Throughput: read all n^2 elements */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_min(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* MAX: result = maximum element                                              */
/* ========================================================================== */

static void iter_max(zap_bencher_t* b, void* param) {
    reduction_ctx_t* ctx = (reduction_ctx_t*)param;

    /* Throughput: read all n^2 elements */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_max(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* NORM_FRO: result = sqrt(sum of squares)                                    */
/* ========================================================================== */

static void iter_norm_fro(zap_bencher_t* b, void* param) {
    reduction_ctx_t* ctx = (reduction_ctx_t*)param;

    /* Throughput: read all n^2 elements */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_norm_fro(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_sum_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("sum");
    zap_group_tag(group, "reduction");

    /* Use vector sizes (n x 1) for consistency with BLAS1 */
    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        reduction_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_vec(n);  /* n x 1 vector */
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("sum", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_sum);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_min_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("min");
    zap_group_tag(group, "reduction");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        reduction_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_vec(n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("min", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_min);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_max_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("max");
    zap_group_tag(group, "reduction");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        reduction_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_vec(n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("max", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_max);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_norm_fro_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("norm_fro");
    zap_group_tag(group, "reduction");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        reduction_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_vec(n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("norm_fro", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_norm_fro);

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

    bench_sum_group();
    bench_min_group();
    bench_max_group();
    bench_norm_fro_group();

    return zap_finalize();
}
