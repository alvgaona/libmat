#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "mat.h"

#define ITERATIONS 100
#define ROUNDS 3

static double ns_per_tick;
Mat *g_m, *g_m2, *g_v1, *g_v2;
volatile float g_sink;
volatile bool g_sink_bool;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

typedef float (*mat_fn)(void);
typedef bool (*mat_fn_bool)(void);

double bench(mat_fn fn) {
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) g_sink = fn();
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < best) best = t;
  }
  return best;
}

double bench_bool(mat_fn_bool fn) {
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) g_sink_bool = fn();
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / ITERATIONS;
    if (t < best) best = t;
  }
  return best;
}

float do_dot_scalar(void) { return mat_dot_scalar_impl(g_v1, g_v2); }
float do_dot_neon(void) { return mat_dot_neon_impl(g_v1, g_v2); }
float do_norm_fro_scalar(void) { return mat_norm_fro_scalar_impl(g_m); }
float do_norm_fro_neon(void) { return mat_norm_fro_neon_impl(g_m); }
float do_norm_max_scalar(void) { return mat_norm_max_scalar_impl(g_m); }
float do_norm_max_neon(void) { return mat_norm_max_neon_impl(g_m); }
float do_nnz_scalar(void) { return mat_nnz_scalar_impl(g_m); }
float do_nnz_neon(void) { return mat_nnz_neon_impl(g_m); }
bool do_equals_scalar(void) { return mat_equals_tol_scalar_impl(g_m, g_m2, 1e-6f); }
bool do_equals_neon(void) { return mat_equals_tol_neon_impl(g_m, g_m2, 1e-6f); }

int main() {
  init_timer();
  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};

  printf("## mat_dot\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
  for (int i = 0; i < 9; i++) {
    int n = sizes[i] * sizes[i];
    g_v1 = mat_ones(n, 1); g_v2 = mat_ones(n, 1);
    double s = bench(do_dot_scalar), ne = bench(do_dot_neon);
    printf("| %d | %.0f ns | %.0f ns | %.1fx |\n", n, s, ne, s/ne);
    mat_free_mat(g_v1); mat_free_mat(g_v2);
  }

  printf("\n## mat_norm_fro\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
  for (int i = 0; i < 9; i++) {
    int sz = sizes[i];
    g_m = mat_ones(sz, sz);
    double s = bench(do_norm_fro_scalar), ne = bench(do_norm_fro_neon);
    printf("| %dx%d | %.0f ns | %.0f ns | %.1fx |\n", sz, sz, s, ne, s/ne);
    mat_free_mat(g_m);
  }

  printf("\n## mat_norm_max\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
  for (int i = 0; i < 9; i++) {
    int sz = sizes[i];
    g_m = mat_ones(sz, sz);
    double s = bench(do_norm_max_scalar), ne = bench(do_norm_max_neon);
    printf("| %dx%d | %.0f ns | %.0f ns | %.1fx |\n", sz, sz, s, ne, s/ne);
    mat_free_mat(g_m);
  }

  printf("\n## mat_nnz\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
  for (int i = 0; i < 9; i++) {
    int sz = sizes[i];
    g_m = mat_ones(sz, sz);
    for (int j = 0; j < sz*sz/2; j += 3) g_m->data[j] = 0;
    double s = bench(do_nnz_scalar), ne = bench(do_nnz_neon);
    printf("| %dx%d | %.0f ns | %.0f ns | %.1fx |\n", sz, sz, s, ne, s/ne);
    mat_free_mat(g_m);
  }

  printf("\n## mat_equals_tol\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");
  for (int i = 0; i < 9; i++) {
    int sz = sizes[i];
    g_m = mat_ones(sz, sz);
    g_m2 = mat_ones(sz, sz);
    for (int j = 0; j < sz*sz; j += 7) g_m2->data[j] += 1e-8f;
    double s = bench_bool(do_equals_scalar), ne = bench_bool(do_equals_neon);
    printf("| %dx%d | %.0f ns | %.0f ns | %.1fx |\n", sz, sz, s, ne, s/ne);
    mat_free_mat(g_m);
    mat_free_mat(g_m2);
  }

  return 0;
}
