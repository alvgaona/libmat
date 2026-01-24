/*
 * ZAP Benchmarks for libmat element-wise operations
 * Operations: abs, sqrt, exp, log, sin, cos
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

/* Fill with values suitable for log (positive values) */
static void fill_positive_mat(Mat* m) {
    size_t n = m->rows * m->cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] = (mat_elem_t)(rand() % 1000) / 100.0f + 0.1f;
    }
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* A;
    Mat* out;
    size_t n;
} elemwise_ctx_t;

/* ========================================================================== */
/* ABS: out = |A|                                                             */
/* ========================================================================== */

static void iter_abs(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_abs(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* SQRT: out = sqrt(A)                                                        */
/* ========================================================================== */

static void iter_sqrt(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_sqrt(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* EXP: out = exp(A)                                                          */
/* ========================================================================== */

static void iter_exp(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_exp(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* LOG: out = log(A)                                                          */
/* ========================================================================== */

static void iter_log(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_log(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* SIN: out = sin(A)                                                          */
/* ========================================================================== */

static void iter_sin(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_sin(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* COS: out = cos(A)                                                          */
/* ========================================================================== */

static void iter_cos(zap_bencher_t* b, void* param) {
    elemwise_ctx_t* ctx = (elemwise_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_cos(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_abs_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("abs");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("abs", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_abs);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_sqrt_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("sqrt");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_positive_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("sqrt", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_sqrt);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_exp_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("exp");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("exp", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_exp);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_log_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("log");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_positive_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("log", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_log);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_sin_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("sin");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("sin", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_sin);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_cos_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("cos");
    zap_group_tag(group, "elementwise");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        elemwise_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("cos", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_cos);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char** argv) {
    srand(42);
    zap_parse_args(argc, argv);

    bench_abs_group();
    bench_sqrt_group();
    bench_exp_group();
    bench_log_group();
    bench_sin_group();
    bench_cos_group();

    return zap_finalize();
}
