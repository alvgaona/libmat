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
  #define LAPACK_GETRF LAPACKE_dgetrf
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_GETRF LAPACKE_sgetrf
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_lu(size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *L = mat_mat(n, n);
  Mat *U = mat_mat(n, n);
  Perm *P = mat_perm(n);

  // Original data
  mat_elem_t *A_orig = malloc(n * n * sizeof(mat_elem_t));
  BENCH_FILL(A_orig, n * n);

  // Make diagonally dominant
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < n; j++) {
      if (i != j) row_sum += (A_orig[i * n + j] > 0 ? A_orig[i * n + j] : -A_orig[i * n + j]);
    }
    A_orig[i * n + i] = row_sum + 1.0f;
  }

  // LAPACK workspace
  mat_elem_t *A_lap = malloc(n * n * sizeof(mat_elem_t));
  lapack_int *ipiv = malloc(n * sizeof(lapack_int));
  lapack_int nn = (lapack_int)n;

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    memcpy(A->data, A_orig, n * n * sizeof(mat_elem_t));
    mat_plu(A, L, U, P);

    memcpy(A_lap, A_orig, n * n * sizeof(mat_elem_t));
    LAPACK_GETRF(LAPACK_ROW_MAJOR, nn, nn, A_lap, nn, ipiv);
  }

  double libmat_times[BENCH_ROUNDS], lapack_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A->data, A_orig, n * n * sizeof(mat_elem_t));
      mat_plu(A, L, U, P);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A_lap, A_orig, n * n * sizeof(mat_elem_t));
      LAPACK_GETRF(LAPACK_ROW_MAJOR, nn, nn, A_lap, nn, ipiv);
    }
    end = bench_now();
    lapack_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(lapack_times, BENCH_ROUNDS);

  // FLOPS for LU: ~2/3 n^3
  double flops = (2.0/3.0) * n * n * n;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_lapack = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.1f ± %.1f us  (%.2fx vs LAPACK)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("LAPACK:   %8.1f ± %.1f us                     %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_lapack);

  mat_free_mat(A);
  mat_free_mat(L);
  mat_free_mat(U);
  mat_free_perm(P);
  free(A_orig);
  free(A_lap);
  free(ipiv);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs LAPACK: LU");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("PA = LU (LU factorization with partial pivoting)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  bench_lu(32, 1000);
  bench_lu(64, 500);
  bench_lu(128, 100);
  bench_lu(256, 20);
  bench_lu(512, 5);
  bench_lu(1024, 2);

  return 0;
}
