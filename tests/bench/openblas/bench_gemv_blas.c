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
  #define BLAS_GEMV cblas_dgemv
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_GEMV cblas_sgemv
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

#define ITERATIONS 1000

void bench_speed(size_t m, size_t n) {
  printf("\n--- Matrix: %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(n);
  Vec *y = mat_vec(m);
  Vec *y_blas = mat_vec(m);

  BENCH_FILL(A->data, m * n);
  BENCH_FILL(x->data, n);
  BENCH_FILL(y->data, m);
  BENCH_FILL(y_blas->data, m);

  mat_elem_t alpha = 1.0, beta = 0.0;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_gemv(y, alpha, A, x, beta);
    BLAS_GEMV(CblasRowMajor, CblasNoTrans, (int)m, (int)n, alpha,
              A->data, (int)n, x->data, 1, beta, y_blas->data, 1);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      mat_gemv(y, alpha, A, x, beta);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < ITERATIONS; i++)
      BLAS_GEMV(CblasRowMajor, CblasNoTrans, (int)m, (int)n, alpha,
                A->data, (int)n, x->data, 1, beta, y_blas->data, 1);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)\n", ls.avg, ls.std, bs.avg / ls.avg);
  printf("OpenBLAS: %8.2f ± %.2f us\n", bs.avg, bs.std);

  mat_free_mat(A);
  mat_free_mat(x);
  mat_free_mat(y);
  mat_free_mat(y_blas);
}

int main() {
  srand(42);
  bench_init();

  printf("=== GEMV BENCHMARK: libmat vs OpenBLAS [%s] ===\n", PRECISION_NAME);
  printf("y = alpha * A * x + beta * y\n");
  printf("Iterations per round: %d\n", ITERATIONS);
  printf("Rounds: %d\n", BENCH_ROUNDS);

  // Square matrices
  bench_speed(64, 64);
  bench_speed(128, 128);
  bench_speed(256, 256);
  bench_speed(512, 512);
  bench_speed(1024, 1024);

  // Tall matrices
  bench_speed(1000, 100);
  bench_speed(10000, 100);

  // Wide matrices
  bench_speed(100, 1000);
  bench_speed(100, 10000);

  return 0;
}
