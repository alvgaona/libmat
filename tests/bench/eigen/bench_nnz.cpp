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

void fill_random_sparse(mat_elem_t *data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    // ~30% sparsity
    data[i] = (rand() % 10 < 3) ? 0.0 : (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
  }
}

volatile mat_elem_t sink;

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  fill_random_sparse(A->data, n);

  Eigen::Map<EigenMatrix> eA(A->data, 1, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_nnz(A);
    sink = (eA.array() != 0).count();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = mat_nnz(A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      sink = (eA.array() != 0).count();
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

  printf("=== NNZ BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("result = count(A != 0)\n");

  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
