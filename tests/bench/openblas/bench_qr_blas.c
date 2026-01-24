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
  #define LAPACK_GEQRF LAPACKE_dgeqrf
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_GEQRF LAPACKE_sgeqrf
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_qr(size_t m, size_t n, int iterations) {
  size_t k = m < n ? m : n;
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Mat *Q = mat_mat(m, m);
  Mat *R = mat_mat(m, n);

  // Original data
  mat_elem_t *A_orig = malloc(m * n * sizeof(mat_elem_t));
  BENCH_FILL(A_orig, m * n);

  // LAPACK workspace
  mat_elem_t *A_lap = malloc(m * n * sizeof(mat_elem_t));
  mat_elem_t *tau = malloc(k * sizeof(mat_elem_t));
  lapack_int mm = (lapack_int)m;
  lapack_int nn = (lapack_int)n;

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    memcpy(A->data, A_orig, m * n * sizeof(mat_elem_t));
    mat_qr(A, Q, R);

    memcpy(A_lap, A_orig, m * n * sizeof(mat_elem_t));
    LAPACK_GEQRF(LAPACK_ROW_MAJOR, mm, nn, A_lap, nn, tau);
  }

  double libmat_times[BENCH_ROUNDS], lapack_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A->data, A_orig, m * n * sizeof(mat_elem_t));
      mat_qr(A, Q, R);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      memcpy(A_lap, A_orig, m * n * sizeof(mat_elem_t));
      LAPACK_GEQRF(LAPACK_ROW_MAJOR, mm, nn, A_lap, nn, tau);
    }
    end = bench_now();
    lapack_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(lapack_times, BENCH_ROUNDS);

  // FLOPS for QR: ~2mn^2 - 2n^3/3 for m >= n
  double flops = 2.0 * m * n * n - (2.0/3.0) * n * n * n;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_lapack = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.1f ± %.1f us  (%.2fx vs LAPACK)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("LAPACK:   %8.1f ± %.1f us                     %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_lapack);

  mat_free_mat(A);
  mat_free_mat(Q);
  mat_free_mat(R);
  free(A_orig);
  free(A_lap);
  free(tau);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs LAPACK: QR");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("A = QR (QR factorization via Householder)\n");
  printf("Note: libmat computes full Q,R; LAPACK geqrf only (compact form)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  printf("\n========== Square matrices ==========\n");
  bench_qr(32, 32, 500);
  bench_qr(64, 64, 200);
  bench_qr(128, 128, 50);
  bench_qr(256, 256, 10);
  bench_qr(512, 512, 2);

  printf("\n========== Tall matrices (m > n) ==========\n");
  bench_qr(128, 64, 100);
  bench_qr(256, 128, 20);
  bench_qr(512, 256, 5);

  return 0;
}
