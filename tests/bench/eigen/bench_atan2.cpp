#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

// atan2 is expensive - use fewer iterations
#define BENCH_ITERATIONS 100
#define BENCH_ROUNDS 10
#define BENCH_WARMUP 3
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenArray = Eigen::ArrayXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenArray = Eigen::ArrayXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_speed(size_t n, int iterations) {
  printf("\n--- Size: %zu ---\n", n);

  Mat *Y = mat_mat(1, n);
  Mat *X = mat_mat(1, n);
  Mat *out = mat_mat(1, n);
  BENCH_FILL(Y->data, n);
  BENCH_FILL(X->data, n);

  Eigen::Map<EigenArray> eY(Y->data, n);
  Eigen::Map<EigenArray> eX(X->data, n);
  EigenArray eOut(n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_atan2(out, Y, X);
    eOut = eY.atan2(eX);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_atan2(out, Y, X);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eOut = eY.atan2(eX);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(Y);
  mat_free_mat(X);
  mat_free_mat(out);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== ATAN2 BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  bench_speed(1000, 100);
  bench_speed(10000, 50);
  bench_speed(100000, 10);
  bench_speed(1000000, 2);

  return 0;
}
