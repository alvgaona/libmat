#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 1000
#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
#else
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
#endif

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Vec *x = mat_vec(n);
  Vec *y = mat_vec(n);
  bench_fill_random_f(x->data, n);
  bench_fill_random_f(y->data, n);

  EigenVector ey = Eigen::Map<EigenVector>(y->data, n);
  Eigen::Map<EigenVector> ex(x->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_axpy(y, alpha, x);
    ey += alpha * ex;
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_axpy(y, alpha, x);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      ey += alpha * ex;
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // read x, read y, write y = 3n
  double gb_libmat = (3.0 * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_eigen = (3.0 * n * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(x); mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== AXPY BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("y += alpha * x\n");

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
