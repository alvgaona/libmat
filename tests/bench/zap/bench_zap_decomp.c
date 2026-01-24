/*
 * ZAP Benchmarks for libmat decomposition operations
 * Operations: qr, plu, chol
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

/* Create a symmetric positive definite matrix: A = M*M^T + n*I */
static void make_spd(Mat* A, size_t n) {
    Mat* M = mat_zeros(n, n);
    fill_random_mat(M);

    /* A = M * M^T */
    mat_syrk(A, M, 1.0f, 0.0f, 'L');

    /* Add n*I to ensure positive definiteness */
    for (size_t i = 0; i < n; i++) {
        MAT_AT(A, i, i) += (mat_elem_t)n;
    }

    /* Fill upper triangle for symmetry */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            MAT_AT(A, i, j) = MAT_AT(A, j, i);
        }
    }

    mat_free_mat(M);
}

/* ========================================================================== */
/* Benchmark contexts                                                         */
/* ========================================================================== */

typedef struct {
    Mat* A;
    Mat* Q;
    Mat* R;
    Mat* L;
    Mat* U;
    Perm* p;
    size_t n;
} decomp_ctx_t;

/* ========================================================================== */
/* QR: A = Q*R                                                                */
/* ========================================================================== */

static void iter_qr(zap_bencher_t* b, void* param) {
    decomp_ctx_t* ctx = (decomp_ctx_t*)param;

    /* Throughput: read A (n^2), write Q (n^2), write R (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_qr(ctx->A, ctx->Q, ctx->R);
        zap_black_box(ctx->Q->data);
        zap_black_box(ctx->R->data);
    });
}

/* ========================================================================== */
/* PLU: P*A = L*U                                                             */
/* ========================================================================== */

static void iter_plu(zap_bencher_t* b, void* param) {
    decomp_ctx_t* ctx = (decomp_ctx_t*)param;

    /* Throughput: read A (n^2), write L (n^2), write U (n^2) */
    zap_bencher_set_throughput_bytes(b, 3 * ctx->n * ctx->n * sizeof(mat_elem_t));

    int result;
    ZAP_ITER(b, {
        result = mat_plu(ctx->A, ctx->L, ctx->U, ctx->p);
        zap_black_box(result);
        zap_black_box(ctx->L->data);
    });
}

/* ========================================================================== */
/* CHOL: A = L*L^T (for SPD matrix)                                           */
/* ========================================================================== */

static void iter_chol(zap_bencher_t* b, void* param) {
    decomp_ctx_t* ctx = (decomp_ctx_t*)param;

    /* Throughput: read A (n^2), write L (n^2/2 lower triangle) */
    zap_bencher_set_throughput_bytes(b, ctx->n * ctx->n * sizeof(mat_elem_t));

    int result;
    ZAP_ITER(b, {
        result = mat_chol(ctx->A, ctx->L);
        zap_black_box(result);
        zap_black_box(ctx->L->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_qr_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("qr");
    zap_group_tag(group, "decomp");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        decomp_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.Q = mat_zeros(n, n);
        ctx.R = mat_zeros(n, n);
        ctx.L = NULL;
        ctx.U = NULL;
        ctx.p = NULL;
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("qr", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_qr);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.Q);
        mat_free_mat(ctx.R);
    }

    zap_group_finish(group);
}

static void bench_plu_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("plu");
    zap_group_tag(group, "decomp");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        decomp_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.Q = NULL;
        ctx.R = NULL;
        ctx.L = mat_zeros(n, n);
        ctx.U = mat_zeros(n, n);
        ctx.p = mat_perm(n);
        fill_random_mat(ctx.A);

        zap_benchmark_id_t id = zap_benchmark_id("plu", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_plu);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.L);
        mat_free_mat(ctx.U);
        mat_free_perm(ctx.p);
    }

    zap_group_finish(group);
}

static void bench_chol_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("chol");
    zap_group_tag(group, "decomp");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        decomp_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.Q = NULL;
        ctx.R = NULL;
        ctx.L = mat_zeros(n, n);
        ctx.U = NULL;
        ctx.p = NULL;

        /* Create SPD matrix for Cholesky */
        make_spd(ctx.A, n);

        zap_benchmark_id_t id = zap_benchmark_id("chol", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_chol);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.L);
    }

    zap_group_finish(group);
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main(int argc, char** argv) {
    srand(42);
    zap_parse_args(argc, argv);

    bench_qr_group();
    bench_plu_group();
    bench_chol_group();

    return zap_finalize();
}
