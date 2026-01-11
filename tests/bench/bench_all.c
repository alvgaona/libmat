// Consolidated benchmark: scalar vs NEON for all optimized functions

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "mat.h"

// ============ Globals ============

static Mat *g_A, *g_B, *g_C;
static Vec *g_x, *g_y;
static volatile mat_elem_t g_sink;
static volatile bool g_sink_b;

#ifdef MAT_HAS_ARM_NEON
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

// ============ Wrappers ============

static void do_dot_s(void) { g_sink = mat_dot_scalar_impl(g_x, g_y); }
static void do_norm_fro_s(void) { g_sink = mat_norm_fro_scalar_impl(g_A); }
static void do_norm_max_s(void) { g_sink = mat_norm_max_scalar_impl(g_A); }
static void do_nnz_s(void) { g_sink = mat_nnz_scalar_impl(g_A); }
static void do_eq_s(void) { g_sink_b = mat_equals_tol_scalar_impl(g_A, g_B, 1e-6f); }
static void do_axpy_s(void) { mat_axpy_scalar_impl(g_y, 2.0, g_x); g_sink = g_y->data[0]; }
static void do_gemv_s(void) { mat_gemv_scalar_impl(g_y, 1.0, g_A, g_x, 0.0); g_sink = g_y->data[0]; }
static void do_ger_s(void) { mat_ger_scalar_impl(g_A, 1.0, g_x, g_y); g_sink = g_A->data[0]; }
static void do_gemm_s(void) { mat_gemm_scalar_impl(g_C, 1.0, g_A, g_B, 0.0); g_sink = g_C->data[0]; }
static void do_t_s(void) { mat_t_scalar_impl(g_B, g_A); g_sink = g_B->data[0]; }

#ifdef MAT_HAS_ARM_NEON
static void do_dot_n(void) { g_sink = mat_dot_neon_impl(g_x, g_y); }
static void do_norm_fro_n(void) { g_sink = mat_norm_fro_neon_impl(g_A); }
static void do_norm_fro_fast_n(void) { g_sink = mat_norm_fro_fast_neon_impl(g_A); }
static void do_norm_max_n(void) { g_sink = mat_norm_max_neon_impl(g_A); }
static void do_nnz_n(void) { g_sink = mat_nnz_neon_impl(g_A); }
static void do_eq_n(void) { g_sink_b = mat_equals_tol_neon_impl(g_A, g_B, 1e-6f); }
static void do_axpy_n(void) { mat_axpy_neon_impl(g_y, 2.0, g_x); g_sink = g_y->data[0]; }
static void do_gemv_n(void) { mat_gemv_neon_impl(g_y, 1.0, g_A, g_x, 0.0); g_sink = g_y->data[0]; }
static void do_ger_n(void) { mat_ger_neon_impl(g_A, 1.0, g_x, g_y); g_sink = g_A->data[0]; }
static void do_gemm_n(void) { mat_gemm_neon_impl(g_C, 1.0, g_A, g_B, 0.0); g_sink = g_C->data[0]; }
static void do_t_n(void) { mat_t_neon_impl(g_B, g_A); g_sink = g_B->data[0]; }
#endif

// ============ Output helpers ============

static void header(const char *name) {
  printf("\n## %s\n\n", name);
#ifdef MAT_HAS_ARM_NEON
  printf("| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
#else
  printf("| Size | Time |\n|------|------|\n");
#endif
}

static void row(const char *size, double scalar, double neon) {
#ifdef MAT_HAS_ARM_NEON
  printf("| %s | %.0f ns | %.0f ns | %.2fx |\n", size, scalar, neon, scalar / neon);
#else
  (void)neon;
  printf("| %s | %.0f ns |\n", size, scalar);
#endif
}

// ============ Benchmark macro ============

#ifdef MAT_HAS_ARM_NEON
#define BENCH(name, scalar_fn, neon_fn, iters) do { \
  double s = bench_run(scalar_fn, iters); \
  double n = bench_run(neon_fn, iters); \
  row(name, s, n); \
} while(0)
#else
#define BENCH(name, scalar_fn, neon_fn, iters) do { \
  double s = bench_run(scalar_fn, iters); \
  row(name, s, 0); \
} while(0)
#endif

// ============ Main ============

int main(void) {
  srand(42);
  bench_init();

  int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
  int n = sizeof(sizes) / sizeof(sizes[0]);
  char sz[32];

  printf("# libmat Benchmark: Scalar vs NEON\n");
  printf("NEON: %s\n", HAS_NEON ? "enabled" : "disabled");

  // mat_dot
  header("mat_dot");
  for (int i = 0; i < n; i++) {
    int len = sizes[i] * sizes[i];
    g_x = mat_mat(len, 1); g_y = mat_mat(len, 1);
    bench_fill_random_f(g_x->data, len);
    bench_fill_random_f(g_y->data, len);
    snprintf(sz, sizeof(sz), "%d", len);
    BENCH(sz, do_dot_s, do_dot_n, BENCH_ITERATIONS);
    mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_norm_fro
  header("mat_norm_fro");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_norm_fro_s, do_norm_fro_n, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

#ifdef MAT_HAS_ARM_NEON
  printf("\n## mat_norm_fro_fast (NEON only)\n\n");
  printf("| Size | Safe | Fast | Speedup |\n|------|------|------|--------|\n");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    double safe = bench_run(do_norm_fro_n, BENCH_ITERATIONS);
    double fast = bench_run(do_norm_fro_fast_n, BENCH_ITERATIONS);
    printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", d, d, safe, fast, safe / fast);
    mat_free_mat(g_A);
  }
#endif

  // mat_norm_max
  header("mat_norm_max");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_norm_max_s, do_norm_max_n, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_nnz
  header("mat_nnz");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    for (int j = 0; j < d * d; j += 3) g_A->data[j] = 0;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_nnz_s, do_nnz_n, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_equals_tol
  header("mat_equals_tol");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    mat_deep_copy(g_B, g_A);
    for (int j = 0; j < d * d; j += 7) g_B->data[j] += 1e-8f;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_eq_s, do_eq_n, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  // mat_axpy
  header("mat_axpy");
  for (int i = 0; i < n; i++) {
    int len = sizes[i] * sizes[i];
    g_x = mat_mat(len, 1); g_y = mat_mat(len, 1);
    bench_fill_random_f(g_x->data, len);
    bench_fill_random_f(g_y->data, len);
    snprintf(sz, sizeof(sz), "%d", len);
    BENCH(sz, do_axpy_s, do_axpy_n, BENCH_ITERATIONS);
    mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_gemv
  header("mat_gemv");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_x = mat_mat(d, 1); g_y = mat_mat(d, 1);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_x->data, d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_gemv_s, do_gemv_n, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_ger
  header("mat_ger");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_x = mat_mat(d, 1); g_y = mat_mat(d, 1);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_x->data, d);
    bench_fill_random_f(g_y->data, d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_ger_s, do_ger_n, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_gemm
  header("mat_gemm");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d); g_C = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_B->data, d * d);
    int iters = (d >= 512) ? 10 : (d >= 256) ? 20 : BENCH_ITERATIONS;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_gemm_s, do_gemm_n, iters);
    mat_free_mat(g_A); mat_free_mat(g_B); mat_free_mat(g_C);
  }

  // mat_t
  header("mat_t");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH(sz, do_t_s, do_t_n, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  printf("\nDone.\n");
  return 0;
}
