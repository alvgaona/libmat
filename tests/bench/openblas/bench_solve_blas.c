#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

// Force single-threaded before OpenBLAS initializes
__attribute__((constructor))
static void force_single_thread(void) {
  setenv("OPENBLAS_NUM_THREADS", "1", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("GOTO_NUM_THREADS", "1", 1);
}

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 5
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define LAPACK_GESV LAPACKE_dgesv
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_GESV LAPACKE_sgesv
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_solve(size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  // Original data
  mat_elem_t *A_orig = malloc(n * n * sizeof(mat_elem_t));
  mat_elem_t *b_orig = malloc(n * sizeof(mat_elem_t));

  // Fill with random data and make diagonally dominant for stability
  BENCH_FILL(A_orig, n * n);
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < n; j++) {
      if (i != j) row_sum += (A_orig[i * n + j] > 0 ? A_orig[i * n + j] : -A_orig[i * n + j]);
    }
    A_orig[i * n + i] = row_sum + 1.0f;
  }
  BENCH_FILL(b_orig, n);

  // LAPACK workspace
  mat_elem_t *A_lap = malloc(n * n * sizeof(mat_elem_t));
  mat_elem_t *x_lap = malloc(n * sizeof(mat_elem_t));
  lapack_int *ipiv = malloc(n * sizeof(lapack_int));
  lapack_int nn = (lapack_int)n;

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    memcpy(A->data, A_orig, n * n * sizeof(mat_elem_t));
    memcpy(b->data, b_orig, n * sizeof(mat_elem_t));
    mat_solve(x, A, b);

    memcpy(A_lap, A_orig, n * n * sizeof(mat_elem_t));
    memcpy(x_lap, b_orig, n * sizeof(mat_elem_t));
    LAPACK_GESV(LAPACK_ROW_MAJOR, nn, 1, A_lap, nn, ipiv, x_lap, 1);
  }

  double libmat_times[BENCH_ROUNDS], lapack_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A->data, A_orig, n * n * sizeof(mat_elem_t));
      memcpy(b->data, b_orig, n * sizeof(mat_elem_t));
      mat_solve(x, A, b);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A_lap, A_orig, n * n * sizeof(mat_elem_t));
      memcpy(x_lap, b_orig, n * sizeof(mat_elem_t));
      LAPACK_GESV(LAPACK_ROW_MAJOR, nn, 1, A_lap, nn, ipiv, x_lap, 1);
    }
    end = bench_now();
    lapack_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(lapack_times, BENCH_ROUNDS);

  // FLOPS for solve: ~2/3 n^3 (LU) + 2n^2 (forward/back sub)
  double flops = (2.0/3.0) * n * n * n + 2.0 * n * n;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_lapack = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.1f ± %.1f us  (%.2fx vs LAPACK)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("LAPACK:   %8.1f ± %.1f us                     %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_lapack);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);
  free(A_orig);
  free(b_orig);
  free(A_lap);
  free(x_lap);
  free(ipiv);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs LAPACK: SOLVE");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("Solve Ax = b (LU with partial pivoting)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  bench_solve(32, 500);
  bench_solve(64, 200);
  bench_solve(128, 50);
  bench_solve(256, 20);
  bench_solve(512, 5);

  return 0;
}
