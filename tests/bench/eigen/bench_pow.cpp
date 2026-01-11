#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

// pow is expensive - use fewer iterations
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
#else
  using EigenArray = Eigen::ArrayXf;
  #define PRECISION_NAME "float32"
#endif

void bench_speed(size_t n, int iterations) {
  printf("\n--- Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  Mat *out = mat_mat(1, n);
  bench_fill_random_f(A->data, n);
  for (size_t i = 0; i < n; i++) A->data[i] = fabsf(A->data[i]) + 0.1f;

  Eigen::Map<EigenArray> eA(A->data, n);
  EigenArray eOut(n);

  mat_elem_t exponent = 2.5f;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_pow(out, A, exponent);
    eOut = eA.pow(exponent);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_pow(out, A, exponent);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eOut = eA.pow(exponent);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(A);
  mat_free_mat(out);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== POW BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  bench_speed(1000, 100);
  bench_speed(10000, 50);
  bench_speed(100000, 10);
  bench_speed(1000000, 2);

  return 0;
}
