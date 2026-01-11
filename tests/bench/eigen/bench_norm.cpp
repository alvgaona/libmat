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
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

volatile mat_elem_t sink;

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Vec *x = mat_vec(n);
  BENCH_FILL(x->data, n);

  Eigen::Map<EigenVector> ex(x->data, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_norm_fro(x);
    sink = mat_norm_fro_fast(x);
    sink = ex.norm();
  }

  double libmat_times[BENCH_ROUNDS], libmat_fast_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_norm_fro(x);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_norm_fro_fast(x);
    end = bench_now();
    libmat_fast_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = ex.norm();
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats lf = bench_stats(libmat_fast_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  double gb_libmat = (n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_fast = (n * sizeof(mat_elem_t)) / (lf.avg * 1000.0);
  double gb_eigen = (n * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat safe: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("libmat fast: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         lf.avg, lf.std, es.avg / lf.avg, gb_fast);
  printf("Eigen:       %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(x);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== NORM BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("result = ||x||_2\n");

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
