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
  #define LAPACK_POTRF LAPACKE_dpotrf
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define LAPACK_POTRF LAPACKE_spotrf
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

// Make matrix SPD: A = B * B^T + n*I
void make_spd(Mat *A) {
  size_t n = A->rows;
  Mat *B = mat_mat(n, n);
  BENCH_FILL(B->data, n * n);

  // A = B * B^T
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= i; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += B->data[i * n + k] * B->data[j * n + k];
      }
      A->data[i * n + j] = sum;
      A->data[j * n + i] = sum;
    }
  }

  // Add n*I
  for (size_t i = 0; i < n; i++) {
    A->data[i * n + i] += (mat_elem_t)n;
  }

  mat_free_mat(B);
}

void bench_chol(size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *L = mat_mat(n, n);
  Mat *A_blas = mat_mat(n, n);

  make_spd(A);

  lapack_int nn = (lapack_int)n;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_chol(A, L);

    mat_deep_copy(A_blas, A);
    LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', nn, A_blas->data, nn);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_chol(A, L);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_deep_copy(A_blas, A);
      LAPACK_POTRF(LAPACK_ROW_MAJOR, 'L', nn, A_blas->data, nn);
    }
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  // FLOPS for Cholesky: ~n^3/3
  double flops = (double)n * n * n / 3.0;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_blas = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs LAPACK)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("LAPACK:   %8.2f ± %.2f us                     %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_blas);

  mat_free_mat(A);
  mat_free_mat(L);
  mat_free_mat(A_blas);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs LAPACK: CHOLESKY");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("A = L * L^T (Cholesky decomposition)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  bench_chol(64, 500);
  bench_chol(128, 200);
  bench_chol(256, 50);
  bench_chol(512, 10);
  bench_chol(1024, 2);

  return 0;
}
