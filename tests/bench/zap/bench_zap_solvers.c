/*
 * ZAP Benchmarks for libmat solver operations
 * Operations: solve, solve_spd
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

static void fill_random_vec(Vec* v) {
    size_t n = v->rows * v->cols;
    for (size_t i = 0; i < n; i++) {
        v->data[i] = (mat_elem_t)(rand() % 1000) / 100.0f;
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

/* Create a diagonally dominant matrix for general solve */
static void make_diag_dominant(Mat* A, size_t n) {
    fill_random_mat(A);

    /* Make diagonally dominant for stability */
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
    Vec* b;
    Vec* x;
    size_t n;
} solver_ctx_t;

/* ========================================================================== */
/* SOLVE: Ax = b (general solver using LU)                                    */
/* ========================================================================== */

static void iter_solve(zap_bencher_t* b, void* param) {
    solver_ctx_t* ctx = (solver_ctx_t*)param;

    /* Throughput: read A (n^2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    ZAP_ITER(b, {
        mat_solve(ctx->x, ctx->A, ctx->b);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* SOLVE_SPD: Ax = b (Cholesky-based solver for SPD matrix)                   */
/* ========================================================================== */

static void iter_solve_spd(zap_bencher_t* b, void* param) {
    solver_ctx_t* ctx = (solver_ctx_t*)param;

    /* Throughput: read A (n^2), read b (n), write x (n) */
    zap_bencher_set_throughput_bytes(b, (ctx->n * ctx->n + 2 * ctx->n) * sizeof(mat_elem_t));

    int result;
    ZAP_ITER(b, {
        result = mat_solve_spd(ctx->x, ctx->A, ctx->b);
        zap_black_box(result);
        zap_black_box(ctx->x->data);
    });
}

/* ========================================================================== */
/* Benchmark groups                                                           */
/* ========================================================================== */

static void bench_solve_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve");
    zap_group_tag(group, "solver");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        solver_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);

        make_diag_dominant(ctx.A, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve);

        mat_free_mat(ctx.A);
        mat_free_mat(ctx.b);
        mat_free_mat(ctx.x);
    }

    zap_group_finish(group);
}

static void bench_solve_spd_group(void) {
    zap_runtime_group_t* group = zap_benchmark_group("solve_spd");
    zap_group_tag(group, "solver");

    /* O(n^3) operations */
    size_t sizes[] = {32, 64, 128, 256, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];

        solver_ctx_t ctx;
        ctx.n = n;
        ctx.A = mat_zeros(n, n);
        ctx.b = mat_vec(n);
        ctx.x = mat_vec(n);

        /* Create SPD matrix for Cholesky-based solver */
        make_spd(ctx.A, n);
        fill_random_vec(ctx.b);

        zap_benchmark_id_t id = zap_benchmark_id("solve_spd", (int64_t)n);
        zap_bench_with_input(group, id, &ctx, sizeof(ctx), iter_solve_spd);

        mat_free_mat(ctx.A);
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

    bench_solve_group();
    bench_solve_spd_group();

    return zap_finalize();
}
