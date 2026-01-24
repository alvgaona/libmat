/*
 * ZAP Benchmarks for libmat advanced operations
 * Operations: inv, svd, det, cond, pinv
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

/* Create a well-conditioned invertible matrix */
static void make_invertible(Mat* A, size_t n) {
    fill_random_mat(A);
    /* Make diagonally dominant for numerical stability */
    for (size_t i = 0; i < n; i++) {
        mat_elem_t row_sum = 0;
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                row_sum += MAT_FABS(MAT_AT(A, i, j));
            }
        }
        MAT_AT(A, i, i) = row_sum + (mat_elem_t)n;
    }
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* A;
    Mat* out;
    Mat* U;
    Vec* S;
    Mat* Vt;
    size_t n;
} advanced_ctx_t;

/* ========================================================================== */
/* INV: out = A^(-1)                                                          */
/* ========================================================================== */

static void iter_inv(zap_bencher_t* b, void* param) {
    advanced_ctx_t* ctx = (advanced_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_inv(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* DET: result = det(A)                                                       */
/* ========================================================================== */

static void iter_det(zap_bencher_t* b, void* param) {
    advanced_ctx_t* ctx = (advanced_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_det(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* SVD: A = U * S * Vt                                                        */
/* ========================================================================== */

static void iter_svd(zap_bencher_t* b, void* param) {
    advanced_ctx_t* ctx = (advanced_ctx_t*)param;

    /* Throughput: read A (n^2), write U (n^2), S (n), Vt (n^2) */
    zap_bencher_set_throughput_bytes(b, (3 * ctx->n * ctx->n + ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_svd(ctx->A, ctx->U, ctx->S, ctx->Vt);
        zap_black_box(ctx->U->data);
        zap_black_box(ctx->S->data);
    });
}

/* ========================================================================== */
/* PINV: out = A^+ (pseudoinverse)                                            */
/* ========================================================================== */

static void iter_pinv(zap_bencher_t* b, void* param) {
    advanced_ctx_t* ctx = (advanced_ctx_t*)param;

    /* Throughput: read A (n^2), write out (n^2) */
    zap_bencher_set_throughput_bytes(b, 2 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_pinv(ctx->out, ctx->A);
        zap_black_box(ctx->out->data);
    });
}

/* ========================================================================== */
/* COND: result = cond(A) (condition number)                                  */
/* ========================================================================== */

static void iter_cond(zap_bencher_t* b, void* param) {
    advanced_ctx_t* ctx = (advanced_ctx_t*)param;

    /* Throughput: read A (n^2) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    mat_elem_t result;
    ZAP_ITER(b, {
        result = mat_cond(ctx->A);
        zap_black_box(result);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_inv_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("inv");
    zap_group_tag(group, "advanced");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        advanced_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        ctx.U = NULL;
        ctx.S = NULL;
        ctx.Vt = NULL;
        make_invertible(ctx.A, n);

        zap_benchmark_id_t id = zap_benchmark_id("inv", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_inv);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_det_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("det");
    zap_group_tag(group, "advanced");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        advanced_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = NULL;
        ctx.U = NULL;
        ctx.S = NULL;
        ctx.Vt = NULL;
        make_invertible(ctx.A, n);

        zap_benchmark_id_t id = zap_benchmark_id("det", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_det);

        mat_free_mat(ctx.A);
    }

    zap_group_finish(group);
}

static void bench_svd_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("svd");
    zap_group_tag(group, "advanced");

    /* SVD is expensive - use smaller sizes */
    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        advanced_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = NULL;
        ctx.U = mat_zeros(n, n);
        ctx.S = mat_vec(n);
        ctx.Vt = mat_zeros(n, n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("svd", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_svd);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.U);
        mat_free_mat(ctx.S);
        mat_free_mat(ctx.Vt);
    }

    zap_group_finish(group);
}

static void bench_pinv_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("pinv");
    zap_group_tag(group, "advanced");

    /* Pseudoinverse uses SVD - expensive */
    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        advanced_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = mat_zeros(n, n);
        ctx.U = NULL;
        ctx.S = NULL;
        ctx.Vt = NULL;
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("pinv", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_pinv);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.out);
    }

    zap_group_finish(group);
}

static void bench_cond_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("cond");
    zap_group_tag(group, "advanced");

    /* Condition number uses SVD */
    size_t sizes[] = {32, 64, 128, 256};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        advanced_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.out = NULL;
        ctx.U = NULL;
        ctx.S = NULL;
        ctx.Vt = NULL;
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("cond", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_cond);

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

    bench_inv_group();
    bench_det_group();
    bench_svd_group();
    bench_pinv_group();
    bench_cond_group();

    return zap_finalize();
}
