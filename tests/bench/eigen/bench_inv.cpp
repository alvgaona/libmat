#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 1
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

void bench_speed(size_t n) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *Ainv = mat_mat(n, n);

  // Make diagonally dominant to ensure invertibility
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < n; j++) {
      mat_elem_t val = (mat_elem_t)(rand() % 100) / 10.0f - 5.0f;
      A->data[i * n + j] = val;
      if (i != j) row_sum += MAT_FABS(val);
    }
    A->data[i * n + i] = row_sum + 1.0f;
  }

  Eigen::Map<EigenMatrix> eA(A->data, n, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_inv(Ainv, A);
    EigenMatrix eAinv = eA.inverse();
    (void)eAinv;
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_inv(Ainv, A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      EigenMatrix eAinv = eA.inverse();
      (void)eAinv;
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f us\n", es.avg, es.std);

  mat_free_mat(A);
  mat_free_mat(Ainv);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== INVERSE BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A^(-1) via LU decomposition\n");

  bench_speed(32);
  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
