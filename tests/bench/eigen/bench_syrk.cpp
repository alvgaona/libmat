#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 10
#define BENCH_ROUNDS 10
#define BENCH_WARMUP 3
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenMatrix = Eigen::MatrixXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_syrk(size_t n, size_t k) {
  printf("\n--- C(%zux%zu) = A(%zux%zu) * A^T (SYRK lower) ---\n", n, n, n, k);

  Mat *A = mat_mat(n, k);
  Mat *C = mat_mat(n, n);
  BENCH_FILL(A->data, n * k);
  mat_fill(C, 0);

  EigenMatrix eA(n, k);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < k; j++)
      eA(i, j) = A->data[i * k + j];
  EigenMatrix eC = EigenMatrix::Zero(n, n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syrk(C, A, 1.0f, 0.0f, 'L');
    eC.selfadjointView<Eigen::Lower>().rankUpdate(eA, 1.0f);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    mat_fill(C, 0);
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_syrk(C, A, 1.0f, 0.0f, 'L');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    eC.setZero();
    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      eC.selfadjointView<Eigen::Lower>().rankUpdate(eA, 1.0f);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPS for SYRK: n^2 * k (approximately)
  double flops = (double)n * n * k;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_eigen = flops / (es.avg * 1000.0);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.1f ± %.1f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A);
  mat_free_mat(C);
}

void bench_syrk_t(size_t n, size_t k) {
  printf("\n--- C(%zux%zu) = A^T(%zux%zu) * A (SYRK_T lower) ---\n", n, n, k, n);

  Mat *A = mat_mat(k, n);  // A is k x n for A^T * A
  Mat *C = mat_mat(n, n);
  BENCH_FILL(A->data, k * n);
  mat_fill(C, 0);

  EigenMatrix eA(k, n);
  for (size_t i = 0; i < k; i++)
    for (size_t j = 0; j < n; j++)
      eA(i, j) = A->data[i * n + j];
  EigenMatrix eC = EigenMatrix::Zero(n, n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syrk_t(C, A, 1.0f, 0.0f, 'L');
    eC.selfadjointView<Eigen::Lower>().rankUpdate(eA.transpose(), 1.0f);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    mat_fill(C, 0);
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_syrk_t(C, A, 1.0f, 0.0f, 'L');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    eC.setZero();
    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      eC.selfadjointView<Eigen::Lower>().rankUpdate(eA.transpose(), 1.0f);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPS for SYRK: n^2 * k (approximately)
  double flops = (double)n * n * k;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_eigen = flops / (es.avg * 1000.0);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.1f ± %.1f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A);
  mat_free_mat(C);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== SYRK BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  printf("\n========== mat_syrk: C = A * A^T ==========\n");
  bench_syrk(64, 64);
  bench_syrk(128, 128);
  bench_syrk(256, 256);
  bench_syrk(512, 512);
  bench_syrk(256, 64);   // tall A
  bench_syrk(256, 512);  // wide A

  printf("\n========== mat_syrk_t: C = A^T * A ==========\n");
  bench_syrk_t(64, 64);
  bench_syrk_t(128, 128);
  bench_syrk_t(256, 256);
  bench_syrk_t(512, 512);
  bench_syrk_t(256, 64);   // A is 64x256
  bench_syrk_t(256, 512);  // A is 512x256

  return 0;
}
