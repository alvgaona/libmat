/*
 * ZAP Benchmarks for libmat BLAS Level 1 operations
 * Operations: axpy, scale, dot, normalize
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

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Vec* x;
    Vec* y;
    size_t n;
} blas1_ctx_t;

/* ========================================================================== */
/* AXPY: y = alpha*x + y                                                      */
/* ========================================================================== */

static void iter_axpy(zap_bencher_t* b, void* param) {
    blas1_ctx_t* ctx = (blas1_ctx_t*)param;
    mat_elem_t alpha = 2.5f;

    /* Throughput: read x (n), read y (n), write y (n) = 3n elements */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_axpy(ctx->y, alpha, ctx->x);
        zap_black_box(ctx->y->data);
    });
}

/* ========================================================================== */
/* SCALE: x = k*x                                                             */
/* ========================================================================== */

static void iter_scale(zap_bencher_t* b, void* param) {
    blas1_ctx_t* ctx = (blas1_ctx_t*)param;
    mat_elem_t k = 1.5f;

    /* Throughput: read x (n), write x (n) = 2n elements */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_scale(ctx->x, k);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* DOT: result = dot(x, y)                                                    */
/* ========================================================================== */

static void iter_dot(zap_bencher_t* b, void* param) {
    blas1_ctx_t* ctx = (blas1_ctx_t*)param;

    /* Throughput: read x (n), read y (n) = 2n elements */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_dot(ctx->x, ctx->y);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* NORMALIZE: normalize vector and return norm                                */
/* ========================================================================== */

static void iter_normalize(zap_bencher_t* b, void* param) {
    blas1_ctx_t* ctx = (blas1_ctx_t*)param;

    /* Throughput: read x (n) for norm, read+write x (2n) for scale = 3n */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * sizeof(mat_elem_t));

    mat_elem_t norm;
    ZAP_ITER(b, {
        /* Refill to avoid degenerate case after repeated normalization */
        fill_random_vec(ctx->x);
        norm = mat_normalize(ctx->x);
        zap_black_box(norm);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_axpy_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("axpy");
    zap_group_tag(group, "blas1");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas1_ctx_t ctx;
        ctx.n = n;
        ctx.x = mat_vec(n);
        ctx.y = mat_vec(n);
        fill_random_vec(ctx.x);
        fill_random_vec(ctx.y);

        zap_benchmark_id_t id = zap_benchmark_id("axpy", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_axpy);

        mat_free_mat(ctx.x);
        mat_free_mat(ctx.y);
    }

    zap_group_finish(group);
}

static void bench_scale_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("scale");
    zap_group_tag(group, "blas1");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas1_ctx_t ctx;
        ctx.n = n;
        ctx.x = mat_vec(n);
        ctx.y = NULL;
        fill_random_vec(ctx.x);

        zap_benchmark_id_t id = zap_benchmark_id("scale", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_scale);

        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

static void bench_dot_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("dot");
    zap_group_tag(group, "blas1");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas1_ctx_t ctx;
        ctx.n = n;
        ctx.x = mat_vec(n);
        ctx.y = mat_vec(n);
        fill_random_vec(ctx.x);
        fill_random_vec(ctx.y);

        zap_benchmark_id_t id = zap_benchmark_id("dot", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_dot);

        mat_free_mat(ctx.x);
        mat_free_mat(ctx.y);
    }

    zap_group_finish(group);
}

static void bench_normalize_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("normalize");
    zap_group_tag(group, "blas1");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        blas1_ctx_t ctx;
        ctx.n = n;
        ctx.x = mat_vec(n);
        ctx.y = NULL;
        fill_random_vec(ctx.x);

        zap_benchmark_id_t id = zap_benchmark_id("normalize", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_normalize);

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

    bench_axpy_group();
    bench_scale_group();
    bench_dot_group();
    bench_normalize_group();

    return zap_finalize();
}
