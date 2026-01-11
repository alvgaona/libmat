#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <mach/mach_time.h>
#include <cblas.h>

#define MAT_IMPLEMENTATION
#include "../../mat.h"

// Select BLAS function based on precision
#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_NRM2 cblas_dnrm2
  #define PRECISION_NAME "float64"
  #define LARGE_VAL 1e200
  #define SMALL_VAL 1e-200
#else
  #define BLAS_NRM2 cblas_snrm2
  #define PRECISION_NAME "float32"
  #define LARGE_VAL 1e30f
  #define SMALL_VAL 1e-30f
#endif

#define ITERATIONS 1000
#define ROUNDS 20
#define WARMUP 10

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

typedef struct {
  double avg;
  double min;
  double std;
} Stats;

Stats compute_stats(double *times, int n) {
  Stats s;
  double sum = 0;
  s.min = DBL_MAX;
  for (int i = 0; i < n; i++) {
    sum += times[i];
    if (times[i] < s.min) s.min = times[i];
  }
  s.avg = sum / n;
  double var = 0;
  for (int i = 0; i < n; i++) {
    var += (times[i] - s.avg) * (times[i] - s.avg);
  }
  s.std = sqrt(var / n);
  return s;
}

void fill_random(Mat *v) {
  for (size_t i = 0; i < v->rows * v->cols; i++) {
    v->data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
  }
}

void bench_speed(size_t size) {
  printf("\n--- Vector size: %zu ---\n", size);

  Mat *v = mat_mat(size, 1);
  fill_random(v);

  volatile mat_elem_t sink;
  for (int i = 0; i < WARMUP; i++) {
    sink = mat_norm2(v);
    sink = mat_norm_fro_fast(v);
    sink = BLAS_NRM2((int)size, v->data, 1);
  }

  double libmat_times[ROUNDS];
  double libmat_fast_times[ROUNDS];
  double blas_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm2(v);
    }
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = mat_norm_fro_fast(v);
    }
    end = mach_absolute_time();
    libmat_fast_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      sink = BLAS_NRM2((int)size, v->data, 1);
    }
    end = mach_absolute_time();
    blas_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  Stats libmat_s = compute_stats(libmat_times, ROUNDS);
  Stats libmat_fast_s = compute_stats(libmat_fast_times, ROUNDS);
  Stats blas_s = compute_stats(blas_times, ROUNDS);

  printf("libmat safe:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n",
         libmat_s.avg, libmat_s.std, blas_s.avg / libmat_s.avg);
  printf("libmat fast:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n",
         libmat_fast_s.avg, libmat_fast_s.std, blas_s.avg / libmat_fast_s.avg);
  printf("OpenBLAS:      %8.2f ± %.2f us\n",
         blas_s.avg, blas_s.std);

  mat_free_mat(v);
}

void bench_precision(void) {
  printf("\n=== PRECISION TESTS ===\n");

  // Normal values
  {
    printf("\n[Normal values: 1.0]\n");
    size_t n = 10000;
    Mat *v = mat_ones(n, 1);
    mat_elem_t safe_r = mat_norm2(v);
    mat_elem_t fast_r = mat_norm_fro_fast(v);
    mat_elem_t blas_r = BLAS_NRM2((int)n, v->data, 1);
    mat_elem_t expected = sqrt((double)n);
    printf("  Expected:     %.10f\n", expected);
    printf("  libmat safe:  %.10f (err: %e)\n", safe_r, fabs(safe_r - expected));
    printf("  libmat fast:  %.10f (err: %e)\n", fast_r, fabs(fast_r - expected));
    printf("  OpenBLAS:     %.10f (err: %e)\n", blas_r, fabs(blas_r - expected));
    mat_free_mat(v);
  }

  // Large values (overflow risk)
  {
    printf("\n[Large values: %.0e]\n", (double)LARGE_VAL);
    size_t n = 100;
    Mat *v = mat_mat(n, 1);
    for (size_t i = 0; i < n; i++) v->data[i] = LARGE_VAL;
    mat_elem_t safe_r = mat_norm2(v);
    mat_elem_t fast_r = mat_norm_fro_fast(v);
    mat_elem_t blas_r = BLAS_NRM2((int)n, v->data, 1);
    mat_elem_t expected = sqrt((double)n) * LARGE_VAL;
    printf("  Expected:     %.6e\n", expected);
    printf("  libmat safe:  %.6e %s\n", safe_r, isinf(safe_r) ? "(OVERFLOW)" : "");
    printf("  libmat fast:  %.6e %s\n", fast_r, isinf(fast_r) ? "(OVERFLOW)" : "");
    printf("  OpenBLAS:     %.6e %s\n", blas_r, isinf(blas_r) ? "(OVERFLOW)" : "");
    mat_free_mat(v);
  }

  // Small values (underflow risk)
  {
    printf("\n[Small values: %.0e]\n", (double)SMALL_VAL);
    size_t n = 100;
    Mat *v = mat_mat(n, 1);
    for (size_t i = 0; i < n; i++) v->data[i] = SMALL_VAL;
    mat_elem_t safe_r = mat_norm2(v);
    mat_elem_t fast_r = mat_norm_fro_fast(v);
    mat_elem_t blas_r = BLAS_NRM2((int)n, v->data, 1);
    mat_elem_t expected = sqrt((double)n) * SMALL_VAL;
    printf("  Expected:     %.6e\n", expected);
    printf("  libmat safe:  %.6e %s\n", safe_r, safe_r == 0 ? "(UNDERFLOW)" : "");
    printf("  libmat fast:  %.6e %s\n", fast_r, fast_r == 0 ? "(UNDERFLOW)" : "");
    printf("  OpenBLAS:     %.6e %s\n", blas_r, blas_r == 0 ? "(UNDERFLOW)" : "");
    mat_free_mat(v);
  }

  // Mixed large and small
  {
    printf("\n[Mixed: one %.0e, rest 1.0]\n", (double)LARGE_VAL);
    size_t n = 100;
    Mat *v = mat_ones(n, 1);
    v->data[0] = LARGE_VAL;
    mat_elem_t safe_r = mat_norm2(v);
    mat_elem_t fast_r = mat_norm_fro_fast(v);
    mat_elem_t blas_r = BLAS_NRM2((int)n, v->data, 1);
    printf("  Expected:     ~%.6e\n", (double)LARGE_VAL);
    printf("  libmat safe:  %.6e %s\n", safe_r, isinf(safe_r) ? "(OVERFLOW)" : "");
    printf("  libmat fast:  %.6e %s\n", fast_r, isinf(fast_r) ? "(OVERFLOW)" : "");
    printf("  OpenBLAS:     %.6e %s\n", blas_r, isinf(blas_r) ? "(OVERFLOW)" : "");
    mat_free_mat(v);
  }
}

int main() {
  srand(42);
  init_timer();

  printf("=== NORM2 BENCHMARK: libmat vs OpenBLAS [%s] ===\n", PRECISION_NAME);
  printf("Iterations per round: %d\n", ITERATIONS);
  printf("Rounds: %d\n", ROUNDS);

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  bench_precision();

  return 0;
}
