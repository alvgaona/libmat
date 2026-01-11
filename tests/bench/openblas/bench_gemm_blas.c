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
  #define BLAS_GEMM cblas_dgemm
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  #define BLAS_GEMM cblas_sgemm
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_speed(size_t m, size_t k, size_t n, int iterations) {
  printf("\n--- %zux%zu * %zux%zu ---\n", m, k, k, n);

  Mat *A = mat_mat(m, k);
  Mat *B = mat_mat(k, n);
  Mat *C = mat_mat(m, n);
  Mat *C_blas = mat_mat(m, n);
  BENCH_FILL(A->data, m * k);
  BENCH_FILL(B->data, k * n);
  BENCH_FILL(C->data, m * n);
  BENCH_FILL(C_blas->data, m * n);

  mat_elem_t alpha = 1.0, beta = 0.0;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_gemm(C, alpha, A, B, beta);
    BLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              (int)m, (int)n, (int)k, alpha,
              A->data, (int)k, B->data, (int)n, beta, C_blas->data, (int)n);
  }

  double libmat_times[BENCH_ROUNDS], blas_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_gemm(C, alpha, A, B, beta);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      BLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k, alpha,
                A->data, (int)k, B->data, (int)n, beta, C_blas->data, (int)n);
    end = bench_now();
    blas_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(blas_times, BENCH_ROUNDS);

  double gflops_libmat = (2.0 * m * n * k) / (ls.avg * 1000.0);
  double gflops_blas = (2.0 * m * n * k) / (bs.avg * 1000.0);

  printf("libmat:   %8.2f ± %.2f us  (%.1fx vs BLAS)  %.1f GFLOPS\n",
         ls.avg, ls.std, bs.avg / ls.avg, gflops_libmat);
  printf("OpenBLAS: %8.2f ± %.2f us                   %.1f GFLOPS\n",
         bs.avg, bs.std, gflops_blas);

  mat_free_mat(A); mat_free_mat(B); mat_free_mat(C); mat_free_mat(C_blas);
}

int main() {
  srand(42);
  bench_init();

  printf("=== GEMM BENCHMARK: libmat vs OpenBLAS [%s] ===\n", PRECISION_NAME);
  printf("C = alpha * A * B + beta * C\n");
  printf("Rounds: %d\n", BENCH_ROUNDS);
  printf("OpenBLAS threads: %d\n", openblas_get_num_threads());

  bench_speed(64, 64, 64, 1000);
  bench_speed(128, 128, 128, 500);
  bench_speed(256, 256, 256, 100);
  bench_speed(512, 512, 512, 20);
  bench_speed(1024, 1024, 1024, 5);

  return 0;
}
