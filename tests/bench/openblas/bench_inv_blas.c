#include <stdio.h>
#include <stdlib.h>
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
  #define LAPACK_GETRI LAPACKE_dgetri
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_GETRF LAPACKE_sgetrf
  #define LAPACK_GETRI LAPACKE_sgetri
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

// Make matrix invertible by adding n to diagonal (diagonally dominant)
static void make_invertible(Mat *A) {
  for (size_t i = 0; i < A->rows; i++) {
    A->data[i * A->cols + i] += (mat_elem_t)A->rows;
  }
}

void bench_inv(size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *A_inv = mat_mat(n, n);
  Mat *A_blas = mat_mat(n, n);
  lapack_int *ipiv = malloc(n * sizeof(lapack_int));

  BENCH_FILL(A->data, n * n);
  make_invertible(A);

  lapack_int nn = (lapack_int)n;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_inv(A_inv, A);

    mat_deep_copy(A_blas, A);
    LAPACK_GETRF(LAPACK_ROW_MAJOR, nn, nn, A_blas->data, nn, ipiv);
    LAPACK_GETRI(LAPACK_ROW_MAJOR, nn, A_blas->data, nn, ipiv);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_inv(A_inv, A);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_deep_copy(A_blas, A);
      LAPACK_GETRF(LAPACK_ROW_MAJOR, nn, nn, A_blas->data, nn, ipiv);
      LAPACK_GETRI(LAPACK_ROW_MAJOR, nn, A_blas->data, nn, ipiv);
    }
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  // FLOPS for LU + inverse: ~2/3 n^3 + 2 n^3 = ~8/3 n^3
  double flops = (8.0 / 3.0) * n * n * n;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_blas = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs LAPACK)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("LAPACK:   %8.2f ± %.2f us                     %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_blas);

  mat_free_mat(A);
  mat_free_mat(A_inv);
  mat_free_mat(A_blas);
  free(ipiv);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs LAPACK: INV");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("A^-1 via LU factorization\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  bench_inv(64, 500);
  bench_inv(128, 200);
  bench_inv(256, 50);
  bench_inv(512, 10);
  bench_inv(1024, 2);

  return 0;
}
