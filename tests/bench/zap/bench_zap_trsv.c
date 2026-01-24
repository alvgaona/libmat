/*
 * ZAP Benchmarks for libmat triangular solvers (TRSV)
 * Operations: solve_tril, solve_triu, solve_trilt, solve_tril_unit
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

/* Create a lower triangular matrix with non-zero diagonal */
static void make_lower_triangular(Mat* L, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            MAT_AT(L, i, j) = (mat_elem_t)(rand() % 1000) / 100.0f + 0.1f;
        }
        /* Ensure diagonal is not too small */
        MAT_AT(L, i, i) = MAT_FABS(MAT_AT(L, i, i)) + 1.0f;
    }
}

/* Create a unit lower triangular matrix (ones on diagonal) */
static void make_unit_lower_triangular(Mat* L, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            MAT_AT(L, i, j) = (mat_elem_t)(rand() % 1000) / 100.0f;
        }
        MAT_AT(L, i, i) = 1.0f;  /* Unit diagonal */
    }
}

/* Create an upper triangular matrix with non-zero diagonal */
static void make_upper_triangular(Mat* U, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            MAT_AT(U, i, j) = (mat_elem_t)(rand() % 1000) / 100.0f + 0.1f;
        }
        /* Ensure diagonal is not too small */
        MAT_AT(U, i, i) = MAT_FABS(MAT_AT(U, i, i)) + 1.0f;
    }
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* T;   /* Triangular matrix */
    Vec* b;
    Vec* x;
    size_t n;
} trsv_ctx_t;

/* ========================================================================== */
/* SOLVE_TRIL: Lx = b (lower triangular)                                      */
/* ========================================================================== */

static void iter_solve_tril(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;

    /* Throughput: read L (n^2/2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n / 2 + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_solve_tril(ctx->x, ctx->T, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* SOLVE_TRIL_UNIT: Lx = b (unit lower triangular)                            */
/* ========================================================================== */

static void iter_solve_tril_unit(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;

    /* Throughput: read L (n^2/2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n / 2 + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_solve_tril_unit(ctx->x, ctx->T, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* SOLVE_TRIU: Ux = b (upper triangular)                                      */
/* ========================================================================== */

static void iter_solve_triu(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;

    /* Throughput: read U (n^2/2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n / 2 + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_solve_triu(ctx->x, ctx->T, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* SOLVE_TRILT: L^T x = b (transposed lower triangular)                       */
/* ========================================================================== */

static void iter_solve_trilt(zap_bencher_t* b, void* param) {
    trsv_ctx_t* ctx = (trsv_ctx_t*)param;

    /* Throughput: read L (n^2/2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n / 2 + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_solve_trilt(ctx->x, ctx->T, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_solve_tril_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve_tril");
    zap_group_tag(group, "trsv");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        trsv_ctx_t ctx;
        ctx.n = n;
        ctx.T = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);
        make_lower_triangular(ctx.T, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve_tril", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve_tril);

        mat_free_mat(ctx.T);
        mat_free_mat(ctx.b);
        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

static void bench_solve_tril_unit_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve_tril_unit");
    zap_group_tag(group, "trsv");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        trsv_ctx_t ctx;
        ctx.n = n;
        ctx.T = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);
        make_unit_lower_triangular(ctx.T, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve_tril_unit", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve_tril_unit);

        mat_free_mat(ctx.T);
        mat_free_mat(ctx.b);
        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

static void bench_solve_triu_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve_triu");
    zap_group_tag(group, "trsv");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        trsv_ctx_t ctx;
        ctx.n = n;
        ctx.T = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);
        make_upper_triangular(ctx.T, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve_triu", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve_triu);

        mat_free_mat(ctx.T);
        mat_free_mat(ctx.b);
        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

static void bench_solve_trilt_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve_trilt");
    zap_group_tag(group, "trsv");

    size_t sizes[] = {32, 64, 128, 256, 512, 1024};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        trsv_ctx_t ctx;
        ctx.n = n;
        ctx.T = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);
        make_lower_triangular(ctx.T, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve_trilt", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve_trilt);

        mat_free_mat(ctx.T);
        mat_free_mat(ctx.b);
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

    bench_solve_tril_group();
    bench_solve_tril_unit_group();
    bench_solve_triu_group();
    bench_solve_trilt_group();

    return zap_finalize();
}
