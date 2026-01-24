#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_SCAL cblas_dscal
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_SCAL cblas_sscal
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

#define ITERATIONS 1000

void bench_speed(size_t n) {
  printf("\n--- Vector size: %zu ---\n", n);

  Vec *x = mat_vec(n);
  Vec *x_blas = mat_vec(n);
  BENCH_FILL(x->data, n);
  BENCH_FILL(x_blas->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_scale(x, alpha);
    BLAS_SCAL((int)n, alpha, x_blas->data, 1);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      mat_scale(x, alpha);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      BLAS_SCAL((int)n, alpha, x_blas->data, 1);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  // Bandwidth: read n + write n = 2n elements
  double gb_libmat = (2.0 * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_blas = (2.0 * n * sizeof(mat_elem_t)) / (bs.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)  %.1f GB/s\n", ls.avg, ls.std, bs.avg / ls.avg, gb_libmat);
  printf("OpenBLAS: %8.2f ± %.2f us                   %.1f GB/s\n", bs.avg, bs.std, gb_blas);

  mat_free_mat(x); mat_free_mat(x_blas);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs OpenBLAS: SCAL");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("x = alpha * x\n");

  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);
  bench_speed(10000000);

  return 0;
}
