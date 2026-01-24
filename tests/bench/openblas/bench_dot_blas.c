#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

// Force single-threaded before OpenBLAS initializes
__attribute__((constructor))
static void force_single_thread(void) {
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("GOTO_NUM_THREADS", "1", 1);
}

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_DOT cblas_ddot
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_DOT cblas_sdot
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

volatile mat_elem_t sink;

void bench_dot(size_t n, int iterations) {
  printf("\n--- Size: %zu ---\n", n);

  Vec *x = mat_vec(n);
  Vec *y = mat_vec(n);
  BENCH_FILL(x->data, n);
  BENCH_FILL(y->data, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_dot(x, y);
    sink = BLAS_DOT((int)n, x->data, 1, y->data, 1);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      sink = mat_dot(x, y);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      sink = BLAS_DOT((int)n, x->data, 1, y->data, 1);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  // Bandwidth: read 2n elements
  double gb_libmat = (2.0 * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_blas = (2.0 * n * sizeof(mat_elem_t)) / (bs.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)  %.1f GB/s\n",
         ls.avg, ls.std, bs.avg / ls.avg, gb_libmat);
  printf("OpenBLAS: %8.2f ± %.2f us                   %.1f GB/s\n",
         bs.avg, bs.std, gb_blas);

  mat_free_mat(x);
  mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs OpenBLAS: DOT");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("result = x . y (dot product)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  bench_dot(100, 100000);
  bench_dot(1000, 50000);
  bench_dot(10000, 10000);
  bench_dot(100000, 1000);
  bench_dot(1000000, 100);

  return 0;
}
