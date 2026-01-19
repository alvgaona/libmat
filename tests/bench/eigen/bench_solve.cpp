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
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_speed(size_t n) {
  printf("\n--- %zux%zu ---\n", n, n);

  // Allocate libmat matrices
  Mat *A = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  // Fill with random data and make diagonally dominant for stability
  BENCH_FILL(A->data, n * n);
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < n; j++) {
      if (i != j) row_sum += (A->data[i * n + j] > 0 ? A->data[i * n + j] : -A->data[i * n + j]);
    }
    A->data[i * n + i] = row_sum + 1.0f;
  }
  BENCH_FILL(b->data, n);

  // Map to Eigen (row-major)
  Eigen::Map<EigenMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
    eA(A->data, n, n, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
  Eigen::Map<EigenVector> eb(b->data, n);
  EigenVector ex(n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve(x, A, b);
    ex = eA.partialPivLu().solve(eb);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_solve(x, A, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      ex = eA.partialPivLu().solve(eb);
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
  mat_free_mat(b);
  mat_free_mat(x);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== SOLVE BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("Solve Ax = b (Partial Pivoting LU)\n");

  bench_speed(32);
  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
