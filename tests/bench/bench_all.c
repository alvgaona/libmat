// Consolidated benchmark: scalar vs NEON for all optimized functions
// Compiles on both ARM (with NEON comparison) and non-ARM (scalar only)

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
static double ns_per_tick;
static void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}
static uint64_t get_time(void) { return mach_absolute_time(); }
static double to_ns(uint64_t start, uint64_t end) { return (end - start) * ns_per_tick; }
#else
static void init_timer(void) {}
static uint64_t get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static double to_ns(uint64_t start, uint64_t end) { return (double)(end - start); }
#endif

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "mat.h"

#define ITERATIONS 100
#define ROUNDS 5
#define WARMUP 3

// Global matrices for benchmarks
static Mat *g_A, *g_B, *g_C;
static Vec *g_x, *g_y;
static volatile mat_elem_t g_sink;
static volatile bool g_sink_bool;

typedef mat_elem_t (*bench_fn)(void);
typedef bool (*bench_fn_bool)(void);

static double bench(bench_fn fn, int iters) {
  for (int i = 0; i < WARMUP; i++) g_sink = fn();

  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = get_time();
    for (int i = 0; i < iters; i++) g_sink = fn();
    uint64_t end = get_time();
    double t = to_ns(start, end) / iters;
    if (t < best) best = t;
  }
  return best;
}

static double bench_bool(bench_fn_bool fn, int iters) {
  for (int i = 0; i < WARMUP; i++) g_sink_bool = fn();

  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = get_time();
    for (int i = 0; i < iters; i++) g_sink_bool = fn();
    uint64_t end = get_time();
    double t = to_ns(start, end) / iters;
    if (t < best) best = t;
  }
  return best;
}

static void fill_random(mat_elem_t *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
}

static void print_header(const char *name) {
  printf("\n## %s\n\n", name);
#ifdef MAT_HAS_ARM_NEON
  printf("| Size | Scalar | NEON | Speedup |\n");
  printf("|------|--------|------|--------|\n");
#else
  printf("| Size | Time |\n");
  printf("|------|------|\n");
#endif
}

static void print_row(const char *size, double scalar_ns, double neon_ns) {
#ifdef MAT_HAS_ARM_NEON
  printf("| %s | %.0f ns | %.0f ns | %.2fx |\n", size, scalar_ns, neon_ns, scalar_ns / neon_ns);
#else
  (void)neon_ns;
  printf("| %s | %.0f ns |\n", size, scalar_ns);
#endif
}

// ============ Benchmark wrappers ============

// mat_dot
static mat_elem_t do_dot_scalar(void) { return mat_dot_scalar_impl(g_x, g_y); }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_dot_neon(void) { return mat_dot_neon_impl(g_x, g_y); }
#endif

// mat_norm_fro
static mat_elem_t do_norm_fro_scalar(void) { return mat_norm_fro_scalar_impl(g_A); }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_norm_fro_neon(void) { return mat_norm_fro_neon_impl(g_A); }
static mat_elem_t do_norm_fro_fast_neon(void) { return mat_norm_fro_fast_neon_impl(g_A); }
#endif

// mat_norm_max
static mat_elem_t do_norm_max_scalar(void) { return mat_norm_max_scalar_impl(g_A); }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_norm_max_neon(void) { return mat_norm_max_neon_impl(g_A); }
#endif

// mat_nnz
static mat_elem_t do_nnz_scalar(void) { return mat_nnz_scalar_impl(g_A); }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_nnz_neon(void) { return mat_nnz_neon_impl(g_A); }
#endif

// mat_equals_tol
static bool do_equals_scalar(void) { return mat_equals_tol_scalar_impl(g_A, g_B, 1e-6f); }
#ifdef MAT_HAS_ARM_NEON
static bool do_equals_neon(void) { return mat_equals_tol_neon_impl(g_A, g_B, 1e-6f); }
#endif

// mat_axpy
static mat_elem_t do_axpy_scalar(void) { mat_axpy_scalar_impl(g_y, 2.0, g_x); return g_y->data[0]; }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_axpy_neon(void) { mat_axpy_neon_impl(g_y, 2.0, g_x); return g_y->data[0]; }
#endif

// mat_gemv
static mat_elem_t do_gemv_scalar(void) { mat_gemv_scalar_impl(g_y, 1.0, g_A, g_x, 0.0); return g_y->data[0]; }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_gemv_neon(void) { mat_gemv_neon_impl(g_y, 1.0, g_A, g_x, 0.0); return g_y->data[0]; }
#endif

// mat_ger
static mat_elem_t do_ger_scalar(void) { mat_ger_scalar_impl(g_A, 1.0, g_x, g_y); return g_A->data[0]; }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_ger_neon(void) { mat_ger_neon_impl(g_A, 1.0, g_x, g_y); return g_A->data[0]; }
#endif

// mat_gemm
static mat_elem_t do_gemm_scalar(void) { mat_gemm_scalar_impl(g_C, 1.0, g_A, g_B, 0.0); return g_C->data[0]; }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_gemm_neon(void) { mat_gemm_neon_impl(g_C, 1.0, g_A, g_B, 0.0); return g_C->data[0]; }
#endif

// mat_t (transpose)
static mat_elem_t do_t_scalar(void) { mat_t_scalar_impl(g_B, g_A); return g_B->data[0]; }
#ifdef MAT_HAS_ARM_NEON
static mat_elem_t do_t_neon(void) { mat_t_neon_impl(g_B, g_A); return g_B->data[0]; }
#endif

// ============ Main ============

int main(void) {
  srand(42);
  init_timer();

  int sizes[] = {16, 32, 64, 128, 256, 512, 1024};
  int nsizes = sizeof(sizes) / sizeof(sizes[0]);
  char size_str[32];

  printf("# libmat Benchmark: Scalar vs NEON\n");
#ifdef MAT_HAS_ARM_NEON
  printf("NEON: enabled\n");
#else
  printf("NEON: disabled (scalar only)\n");
#endif
#ifdef MAT_DOUBLE_PRECISION
  printf("Precision: float64\n");
#else
  printf("Precision: float32\n");
#endif

  // -------- mat_dot --------
  print_header("mat_dot");
  for (int i = 0; i < nsizes; i++) {
    int n = sizes[i] * sizes[i];
    g_x = mat_mat(n, 1); g_y = mat_mat(n, 1);
    fill_random(g_x->data, n); fill_random(g_y->data, n);

    double s = bench(do_dot_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_dot_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%d", n);
    print_row(size_str, s, ne);

    mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // -------- mat_norm_fro --------
  print_header("mat_norm_fro");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);

    double s = bench(do_norm_fro_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_norm_fro_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A);
  }

#ifdef MAT_HAS_ARM_NEON
  // -------- mat_norm_fro_fast (NEON only) --------
  printf("\n## mat_norm_fro_fast (NEON only, no overflow protection)\n\n");
  printf("| Size | Safe NEON | Fast NEON | Speedup |\n");
  printf("|------|-----------|-----------|--------|\n");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);

    double safe = bench(do_norm_fro_neon, ITERATIONS);
    double fast = bench(do_norm_fro_fast_neon, ITERATIONS);
    printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", sz, sz, safe, fast, safe / fast);

    mat_free_mat(g_A);
  }
#endif

  // -------- mat_norm_max --------
  print_header("mat_norm_max");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);

    double s = bench(do_norm_max_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_norm_max_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A);
  }

  // -------- mat_nnz --------
  print_header("mat_nnz");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);
    // Make ~30% zeros
    for (int j = 0; j < sz * sz; j += 3) g_A->data[j] = 0;

    double s = bench(do_nnz_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_nnz_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A);
  }

  // -------- mat_equals_tol --------
  print_header("mat_equals_tol");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    g_B = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);
    mat_deep_copy(g_B, g_A);
    // Add small noise
    for (int j = 0; j < sz * sz; j += 7) g_B->data[j] += 1e-8f;

    double s = bench_bool(do_equals_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench_bool(do_equals_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  // -------- mat_axpy --------
  print_header("mat_axpy");
  for (int i = 0; i < nsizes; i++) {
    int n = sizes[i] * sizes[i];
    g_x = mat_mat(n, 1); g_y = mat_mat(n, 1);
    fill_random(g_x->data, n); fill_random(g_y->data, n);

    double s = bench(do_axpy_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_axpy_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%d", n);
    print_row(size_str, s, ne);

    mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // -------- mat_gemv --------
  print_header("mat_gemv");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    g_x = mat_mat(sz, 1);
    g_y = mat_mat(sz, 1);
    fill_random(g_A->data, sz * sz);
    fill_random(g_x->data, sz);

    double s = bench(do_gemv_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_gemv_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // -------- mat_ger --------
  print_header("mat_ger");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    g_x = mat_mat(sz, 1);
    g_y = mat_mat(sz, 1);
    fill_random(g_A->data, sz * sz);
    fill_random(g_x->data, sz);
    fill_random(g_y->data, sz);

    double s = bench(do_ger_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_ger_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A); mat_free_mat(g_x); mat_free_mat(g_y);
  }

  // -------- mat_gemm --------
  print_header("mat_gemm");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    g_B = mat_mat(sz, sz);
    g_C = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);
    fill_random(g_B->data, sz * sz);

    // Fewer iterations for large GEMM
    int iters = (sz >= 512) ? 10 : (sz >= 256) ? 20 : ITERATIONS;

    double s = bench(do_gemm_scalar, iters);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_gemm_neon, iters);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A); mat_free_mat(g_B); mat_free_mat(g_C);
  }

  // -------- mat_t (transpose) --------
  print_header("mat_t (transpose)");
  for (int i = 0; i < nsizes; i++) {
    int sz = sizes[i];
    g_A = mat_mat(sz, sz);
    g_B = mat_mat(sz, sz);
    fill_random(g_A->data, sz * sz);

    double s = bench(do_t_scalar, ITERATIONS);
#ifdef MAT_HAS_ARM_NEON
    double ne = bench(do_t_neon, ITERATIONS);
#else
    double ne = 0;
#endif
    snprintf(size_str, sizeof(size_str), "%dx%d", sz, sz);
    print_row(size_str, s, ne);

    mat_free_mat(g_A); mat_free_mat(g_B);
  }

  printf("\nDone.\n");
  return 0;
}
