/*
 * ZAP Benchmarks for libmat norm operations
 * Operations: norm2, norm_max, norm_fro_fast, norm (Lp)
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
} norm_ctx_t;

/* ========================================================================== */
/* NORM2: result = ||A||_2 (Frobenius)                                        */
/* ========================================================================== */

static void iter_norm2(zap_bencher_t* b, void* param) {
    norm_ctx_t* ctx = (norm_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_norm2(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* NORM_MAX: result = ||A||_inf                                               */
/* ========================================================================== */

static void iter_norm_max(zap_bencher_t* b, void* param) {
    norm_ctx_t* ctx = (norm_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_norm_max(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* NORM_FRO_FAST: result = ||A||_F (fast approximation)                       */
/* ========================================================================== */

static void iter_norm_fro_fast(zap_bencher_t* b, void* param) {
    norm_ctx_t* ctx = (norm_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_norm_fro_fast(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* NORM_L1: result = ||A||_1                                                  */
/* ========================================================================== */

static void iter_norm_l1(zap_bencher_t* b, void* param) {
    norm_ctx_t* ctx = (norm_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_norm(ctx->A, 1);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_norm2_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("norm2");
    zap_group_tag(group, "norms");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        norm_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("norm2", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_norm2);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_norm_max_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("norm_max");
    zap_group_tag(group, "norms");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        norm_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("norm_max", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_norm_max);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_norm_fro_fast_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("norm_fro_fast");
    zap_group_tag(group, "norms");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        norm_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("norm_fro_fast", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_norm_fro_fast);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_norm_l1_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("norm_l1");
    zap_group_tag(group, "norms");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        norm_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("norm_l1", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_norm_l1);

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

    bench_norm2_group();
    bench_norm_max_group();
    bench_norm_fro_fast_group();
    bench_norm_l1_group();

    return zap_finalize();
}
