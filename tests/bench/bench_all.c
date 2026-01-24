// Consolidated benchmark: compare all architectures for optimized functions

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

// ============ Wrapper functions by architecture ============
// Note: Internal functions use trailing underscore (e.g., mat_gemm_scalar_)

// Scalar implementations (always available)
static void do_norm_fro_scalar(void) { g_sink = mat_norm_fro_scalar_(g_A); }
static void do_nnz_scalar(void) { g_sink = mat_nnz_scalar_(g_A); }
static void do_eq_scalar(void) { g_sink_b = mat_equals_tol_scalar_(g_A, g_B, 1e-6f); }
static void do_gemv_scalar(void) { mat_gemv_scalar_(g_y, 1.0, g_A, g_x, 0.0); g_sink = g_y->data[0]; }
static void do_ger_scalar(void) { mat_outer_scalar_(g_A, g_x, g_y); g_sink = g_A->data[0]; }
static void do_gemm_scalar(void) { mat_gemm_scalar_(g_C, 1.0, g_A, g_B, 0.0); g_sink = g_C->data[0]; }
static void do_t_scalar(void) { mat_t_scalar_(g_B, g_A); g_sink = g_B->data[0]; }
static void do_sum_scalar(void) { g_sink = mat_sum_scalar_(g_A); }
static void do_min_scalar(void) { g_sink = mat_min_scalar_(g_A); }
static void do_max_scalar(void) { g_sink = mat_max_scalar_(g_A); }

// NEON implementations
#ifdef MAT_HAS_ARM_NEON
static void do_norm_fro_neon(void) { g_sink = mat_norm_fro_neon_(g_A); }
static void do_norm_fro_fast_neon(void) { g_sink = mat_norm_fro_fast_neon_(g_A); }
static void do_nnz_neon(void) { g_sink = mat_nnz_neon_(g_A); }
static void do_eq_neon(void) { g_sink_b = mat_equals_tol_neon_(g_A, g_B, 1e-6f); }
static void do_gemv_neon(void) { mat_gemv_neon_(g_y, 1.0, g_A, g_x, 0.0); g_sink = g_y->data[0]; }
static void do_ger_neon(void) { mat_outer_neon_(g_A, g_x, g_y); g_sink = g_A->data[0]; }
static void do_gemm_neon(void) { mat_gemm_neon_(g_C, 1.0, g_A, g_B, 0.0); g_sink = g_C->data[0]; }
static void do_t_neon(void) { mat_t_neon_(g_B, g_A); g_sink = g_B->data[0]; }
static void do_sum_neon(void) { g_sink = mat_sum_neon_(g_A); }
static void do_min_neon(void) { g_sink = mat_min_neon_(g_A); }
static void do_max_neon(void) { g_sink = mat_max_neon_(g_A); }
#else
#define do_norm_fro_neon NULL
#define do_norm_fro_fast_neon NULL
#define do_nnz_neon NULL
#define do_eq_neon NULL
#define do_gemv_neon NULL
#define do_ger_neon NULL
#define do_gemm_neon NULL
#define do_t_neon NULL
#define do_sum_neon NULL
#define do_min_neon NULL
#define do_max_neon NULL
#endif

// AVX2 implementations (future)
#define do_norm_fro_avx2 NULL
#define do_nnz_avx2 NULL
#define do_eq_avx2 NULL
#define do_gemv_avx2 NULL
#define do_ger_avx2 NULL
#define do_gemm_avx2 NULL
#define do_t_avx2 NULL
#define do_sum_avx2 NULL
#define do_min_avx2 NULL
#define do_max_avx2 NULL

// ============ Generic benchmark macro ============

#define BENCH_OP(size_str, scalar, neon, avx2, iters) do { \
  double _t[BENCH_ARCH_MAX] = {0}; \
  bench_void_fn _f[] = {scalar, neon, avx2}; \
  for (int _i = 0; _i < BENCH_ARCH_MAX; _i++) \
    if (bench_arch_available[_i] && _f[_i]) \
      _t[_i] = bench_run(_f[_i], iters); \
  bench_print_row(size_str, _t); \
} while(0)

// ============ Main ============

int main(void) {
  srand(42);
  bench_init();

  int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
  int n = sizeof(sizes) / sizeof(sizes[0]);
  char sz[32];

  bench_print_summary("libmat Internal Benchmark");

  // mat_norm_fro
  bench_print_header("mat_norm_fro");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_norm_fro_scalar, do_norm_fro_neon, do_norm_fro_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

#ifdef MAT_HAS_ARM_NEON
  printf("\n## mat_norm_fro_fast (NEON only)\n\n");
  printf("| Size | Safe | Fast | Speedup |\n|------|------|------|--------|\n");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    double safe = bench_run(do_norm_fro_neon, BENCH_ITERATIONS);
    double fast = bench_run(do_norm_fro_fast_neon, BENCH_ITERATIONS);
    printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", d, d, safe, fast, safe / fast);
    mat_free_mat(g_A);
  }
#endif

  // mat_sum
  printf("\n");
  bench_print_header("mat_sum");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_sum_scalar, do_sum_neon, do_sum_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_min
  printf("\n");
  bench_print_header("mat_min");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_min_scalar, do_min_neon, do_min_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_max
  printf("\n");
  bench_print_header("mat_max");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_max_scalar, do_max_neon, do_max_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_nnz
  printf("\n");
  bench_print_header("mat_nnz");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    for (int j = 0; j < d * d; j += 3) g_A->data[j] = 0;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_nnz_scalar, do_nnz_neon, do_nnz_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A);
  }

  // mat_equals_tol
  printf("\n");
  bench_print_header("mat_equals_tol");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    mat_deep_copy(g_B, g_A);
    for (int j = 0; j < d * d; j += 7) g_B->data[j] += 1e-8f;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_eq_scalar, do_eq_neon, do_eq_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  // mat_gemv
  printf("\n");
  bench_print_header("mat_gemv");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_x = mat_mat(d, 1); g_y = mat_mat(d, 1);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_x->data, d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_gemv_scalar, do_gemv_neon, do_gemv_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_ger (outer product)
  printf("\n");
  bench_print_header("mat_ger");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_x = mat_mat(d, 1); g_y = mat_mat(d, 1);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_x->data, d);
    bench_fill_random_f(g_y->data, d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_ger_scalar, do_ger_neon, do_ger_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // mat_gemm
  printf("\n");
  bench_print_header("mat_gemm");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d); g_C = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    bench_fill_random_f(g_B->data, d * d);
    int iters = (d >= 512) ? 10 : (d >= 256) ? 20 : BENCH_ITERATIONS;
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_gemm_scalar, do_gemm_neon, do_gemm_avx2, iters);
    mat_free_mat(g_A); mat_free_mat(g_B); mat_free_mat(g_C);
  }

  // mat_t
  printf("\n");
  bench_print_header("mat_t");
  for (int i = 0; i < n; i++) {
    int d = sizes[i];
    g_A = mat_mat(d, d); g_B = mat_mat(d, d);
    bench_fill_random_f(g_A->data, d * d);
    snprintf(sz, sizeof(sz), "%dx%d", d, d);
    BENCH_OP(sz, do_t_scalar, do_t_neon, do_t_avx2, BENCH_ITERATIONS);
    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  printf("\nDone.\n");
  return 0;
}
