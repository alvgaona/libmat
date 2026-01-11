#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 5
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenMatrix = Eigen::MatrixXd;
  #define PRECISION_NAME "float64"
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
#endif

void bench_speed(size_t m, size_t k, size_t n, int iterations) {
  printf("\n--- %zux%zu * %zux%zu ---\n", m, k, k, n);

  // libmat
  Mat *A = mat_mat(m, k);
  Mat *B = mat_mat(k, n);
  Mat *C = mat_mat(m, n);
  bench_fill_random_f(A->data, m * k);
  bench_fill_random_f(B->data, k * n);

  // Eigen (map to same data for fair comparison)
  Eigen::Map<EigenMatrix> eA(A->data, m, k);
  Eigen::Map<EigenMatrix> eB(B->data, k, n);
  EigenMatrix eC(m, n);

  mat_elem_t alpha = 1.0, beta = 0.0;

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_gemm(C, alpha, A, B, beta);
    eC.noalias() = eA * eB;
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_gemm(C, alpha, A, B, beta);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eC.noalias() = eA * eB;
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  double gflops_libmat = (2.0 * m * n * k) / (ls.avg * 1000.0);
  double gflops_eigen = (2.0 * m * n * k) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A); mat_free_mat(B); mat_free_mat(C);
}

int main() {
  srand(42);
  bench_init();

  // Disable Eigen's parallelization for fair comparison
  Eigen::setNbThreads(1);

  printf("=== GEMM BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("C = A * B\n");
  printf("Eigen threads: %d\n", Eigen::nbThreads());

  bench_speed(64, 64, 64, 1000);
  bench_speed(128, 128, 128, 500);
  bench_speed(256, 256, 256, 100);
  bench_speed(512, 512, 512, 20);
  bench_speed(1024, 1024, 1024, 5);

  return 0;
}
