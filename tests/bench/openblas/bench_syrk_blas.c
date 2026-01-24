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
#define BENCH_WARMUP 5
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  #define BLAS_SYRK cblas_dsyrk
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_SYRK cblas_ssyrk
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_syrk(size_t n, size_t k, int iterations) {
  printf("\n--- C(%zux%zu) = A(%zux%zu) * A^T ---\n", n, n, n, k);

  Mat *A = mat_mat(n, k);
  Mat *C = mat_mat(n, n);
  Mat *C_blas = mat_mat(n, n);
  BENCH_FILL(A->data, n * k);
  mat_fill(C, 0);
  mat_fill(C_blas, 0);

  mat_elem_t alpha = 1.0, beta = 0.0;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syrk(C, A, alpha, beta, 'L');
    BLAS_SYRK(CblasRowMajor, CblasLower, CblasNoTrans,
              (int)n, (int)k, alpha, A->data, (int)k, beta, C_blas->data, (int)n);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    mat_fill(C, 0);
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_syrk(C, A, alpha, beta, 'L');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    mat_fill(C_blas, 0);
    start = bench_now();
    for (int i = 0; i < iterations; i++)
      BLAS_SYRK(CblasRowMajor, CblasLower, CblasNoTrans,
                (int)n, (int)k, alpha, A->data, (int)k, beta, C_blas->data, (int)n);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  // FLOPS for SYRK: ~n^2 * k
  double flops = (double)n * n * k;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_blas = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.1f ± %.1f us  (%.2fx vs BLAS)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("OpenBLAS: %8.1f ± %.1f us                   %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_blas);

  mat_free_mat(A);
  mat_free_mat(C);
  mat_free_mat(C_blas);
}

void bench_syrk_t(size_t n, size_t k, int iterations) {
  printf("\n--- C(%zux%zu) = A^T(%zux%zu) * A ---\n", n, n, k, n);

  Mat *A = mat_mat(k, n);  // A is k x n for A^T * A
  Mat *C = mat_mat(n, n);
  Mat *C_blas = mat_mat(n, n);
  BENCH_FILL(A->data, k * n);
  mat_fill(C, 0);
  mat_fill(C_blas, 0);

  mat_elem_t alpha = 1.0, beta = 0.0;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syrk_t(C, A, alpha, beta, 'L');
    BLAS_SYRK(CblasRowMajor, CblasLower, CblasTrans,
              (int)n, (int)k, alpha, A->data, (int)n, beta, C_blas->data, (int)n);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    mat_fill(C, 0);
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_syrk_t(C, A, alpha, beta, 'L');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    mat_fill(C_blas, 0);
    start = bench_now();
    for (int i = 0; i < iterations; i++)
      BLAS_SYRK(CblasRowMajor, CblasLower, CblasTrans,
                (int)n, (int)k, alpha, A->data, (int)n, beta, C_blas->data, (int)n);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  double flops = (double)n * n * k;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_blas = flops / (bs.avg * 1000.0);

  printf("libmat:   %8.1f ± %.1f us  (%.2fx vs BLAS)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("OpenBLAS: %8.1f ± %.1f us                   %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_blas);

  mat_free_mat(A);
  mat_free_mat(C);
  mat_free_mat(C_blas);
}

int main() {
  srand(42);
  bench_init();

  bench_print_summary("libmat vs OpenBLAS: SYRK");
  printf("Precision: %s\n", PRECISION_NAME);
  printf("C = alpha * A * A^T + beta * C (symmetric rank-k update)\n");
  printf("Rounds: %d, OpenBLAS threads: %d\n", BENCH_ROUNDS, openblas_get_num_threads());

  printf("\n========== mat_syrk: C = A * A^T ==========\n");
  bench_syrk(64, 64, 500);
  bench_syrk(128, 128, 200);
  bench_syrk(256, 256, 50);
  bench_syrk(512, 512, 10);

  printf("\n========== mat_syrk_t: C = A^T * A ==========\n");
  bench_syrk_t(64, 64, 500);
  bench_syrk_t(128, 128, 200);
  bench_syrk_t(256, 256, 50);
  bench_syrk_t(512, 512, 10);

  return 0;
}
