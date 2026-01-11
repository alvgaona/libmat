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
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

volatile mat_elem_t sink;

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  BENCH_FILL(A->data, n);

  Eigen::Map<EigenMatrix> eA(A->data, 1, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_norm_max(A);
    sink = eA.lpNorm<Eigen::Infinity>();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_norm_max(A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = eA.lpNorm<Eigen::Infinity>();
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

  printf("=== NORM_MAX BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("result = max(|A|) (infinity norm)\n");

  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
