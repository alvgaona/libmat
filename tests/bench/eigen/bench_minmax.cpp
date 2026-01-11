#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 10000
#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
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

volatile mat_elem_t sink;

void bench_min(size_t n) {
  printf("\n--- MIN Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  bench_fill_random_f(A->data, n);

  Eigen::Map<EigenMatrix> eA(A->data, 1, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_min(A);
    sink = eA.minCoeff();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_min(A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = eA.minCoeff();
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.3f ± %.3f us  (%.1fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.3f ± %.3f us\n", es.avg, es.std);

  mat_free_mat(A);
}

void bench_max(size_t n) {
  printf("\n--- MAX Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  bench_fill_random_f(A->data, n);

  Eigen::Map<EigenMatrix> eA(A->data, 1, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_max(A);
    sink = eA.maxCoeff();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_max(A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = eA.maxCoeff();
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.3f ± %.3f us  (%.1fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.3f ± %.3f us\n", es.avg, es.std);

  mat_free_mat(A);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== MIN/MAX BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  bench_min(1000);
  bench_min(10000);
  bench_min(100000);
  bench_min(1000000);

  bench_max(1000);
  bench_max(10000);
  bench_max(100000);
  bench_max(1000000);

  return 0;
}
