/*
 * ZAP Benchmarks for libmat matrix operations
 * Operations: transpose, add, sub, hadamard, mul
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
} matop_ctx_t;

/* ========================================================================== */
/* TRANSPOSE: C = A^T                                                         */
/* ========================================================================== */

static void iter_transpose(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_t(ctx->C, ctx->A);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* ADD: C = A + B                                                             */
/* ========================================================================== */

static void iter_add(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), read B (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_add(ctx->C, ctx->A, ctx->B);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* SUB: C = A - B                                                             */
/* ========================================================================== */

static void iter_sub(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), read B (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_sub(ctx->C, ctx->A, ctx->B);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* HADAMARD: C = A .* B (element-wise multiply)                               */
/* ========================================================================== */

static void iter_hadamard(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), read B (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_hadamard(ctx->C, ctx->A, ctx->B);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* MUL: C = A * B (matrix multiply via mat_mul)                               */
/* ========================================================================== */

static void iter_mul(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), read B (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_mul(ctx->C, ctx->A, ctx->B);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* DEEP_COPY: C = copy(A)                                                     */
/* ========================================================================== */

static void iter_deep_copy(zap_bencher_t* b, void* param) {
    matop_ctx_t* ctx = (matop_ctx_t*)param;

    /* Throughput: read A (n^2), write C (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_deep_copy(ctx->C, ctx->A);
        zap_black_box(ctx->C->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_transpose_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("transpose");
    zap_group_tag(group, "matop");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = NULL;
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("transpose", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_transpose);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_add_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("add");
    zap_group_tag(group, "matop");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = mat_zeros(n, n);
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);
        fill_random_mat(ctx.B);

        zap_benchmark_id_t id = zap_benchmark_id("add", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_add);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.B);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_sub_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("sub");
    zap_group_tag(group, "matop");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = mat_zeros(n, n);
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);
        fill_random_mat(ctx.B);

        zap_benchmark_id_t id = zap_benchmark_id("sub", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_sub);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.B);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_hadamard_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("hadamard");
    zap_group_tag(group, "matop");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = mat_zeros(n, n);
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);
        fill_random_mat(ctx.B);

        zap_benchmark_id_t id = zap_benchmark_id("hadamard", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_hadamard);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.B);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_mul_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("mul");
    zap_group_tag(group, "matop");

    /* O(n^3) - use smaller sizes */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = mat_zeros(n, n);
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);
        fill_random_mat(ctx.B);

        zap_benchmark_id_t id = zap_benchmark_id("mul", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_mul);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.B);
        mat_free_mat(ctx.C);
    }

    zap_group_finish(group);
}

static void bench_deep_copy_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("deep_copy");
    zap_group_tag(group, "matop");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        matop_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.B = NULL;
        ctx.C = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("deep_copy", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_deep_copy);

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

    bench_transpose_group();
    bench_add_group();
    bench_sub_group();
    bench_hadamard_group();
    bench_mul_group();
    bench_deep_copy_group();

    return zap_finalize();
}
