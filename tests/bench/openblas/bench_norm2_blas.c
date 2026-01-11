#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_NRM2 cblas_dnrm2
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
  #define LARGE_VAL 1e200
  #define SMALL_VAL 1e-200
#else
  #define BLAS_NRM2 cblas_snrm2
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
  #define LARGE_VAL 1e30f
  #define SMALL_VAL 1e-30f
#endif

#define ITERATIONS 1000

void bench_speed(size_t size) {
  printf("\n--- Vector size: %zu ---\n", size);

  Mat *v = mat_mat(size, 1);
  BENCH_FILL(v->data, size);

  volatile mat_elem_t sink;
  (void)sink;
  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_norm2(v);
    sink = mat_norm_fro_fast(v);
    sink = BLAS_NRM2((int)size, v->data, 1);
  }

  double libmat_times[BENCH_ROUNDS];
  double libmat_fast_times[BENCH_ROUNDS];
  double blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      sink = mat_norm2(v);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      sink = mat_norm_fro_fast(v);
    end = bench_now();
    libmat_fast_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      sink = BLAS_NRM2((int)size, v->data, 1);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats lfs = bench_stats(libmat_fast_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  printf("libmat safe:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n",
         ls.avg, ls.std, bs.avg / ls.avg);
  printf("libmat fast:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n",
         lfs.avg, lfs.std, bs.avg / lfs.avg);
  printf("OpenBLAS:      %8.2f ± %.2f us\n",
         bs.avg, bs.std);

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
  bench_init();

  printf("=== NORM2 BENCHMARK: libmat vs OpenBLAS [%s] ===\n", PRECISION_NAME);
  printf("Iterations per round: %d\n", ITERATIONS);
  printf("Rounds: %d\n", BENCH_ROUNDS);
  printf("OpenBLAS threads: %d\n", openblas_get_num_threads());

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  bench_precision();

  return 0;
}
