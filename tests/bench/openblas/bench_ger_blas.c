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
  #define BLAS_GER cblas_dger
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_GER cblas_sger
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

#define ITERATIONS 1000

void bench_speed(size_t m, size_t n) {
  printf("\n--- Matrix: %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Mat *A_blas = mat_mat(m, n);
  Vec *x = mat_vec(m);
  Vec *y = mat_vec(n);
  BENCH_FILL(A->data, m * n);
  BENCH_FILL(A_blas->data, m * n);
  BENCH_FILL(x->data, m);
  BENCH_FILL(y->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_ger(A, alpha, x, y);
    BLAS_GER(CblasRowMajor, (int)m, (int)n, alpha, x->data, 1, y->data, 1, A_blas->data, (int)n);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      mat_ger(A, alpha, x, y);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      BLAS_GER(CblasRowMajor, (int)m, (int)n, alpha, x->data, 1, y->data, 1, A_blas->data, (int)n);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);
  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n", ls.avg, ls.std, bs.avg / ls.avg);
  printf("OpenBLAS: %8.2f ± %.2f us\n", bs.avg, bs.std);

  mat_free_mat(A); mat_free_mat(A_blas); mat_free_mat(x); mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs OpenBLAS: GER");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("A += alpha * x * y^T\n\n");

  bench_speed(64, 64);
  bench_speed(128, 128);
  bench_speed(256, 256);
  bench_speed(512, 512);
  bench_speed(1024, 1024);

  return 0;
}
