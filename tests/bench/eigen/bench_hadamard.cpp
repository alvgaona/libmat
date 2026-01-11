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
  using EigenMatrix = Eigen::MatrixXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Mat *A = mat_mat(1, n);
  Mat *B = mat_mat(1, n);
  Mat *C = mat_mat(1, n);
  BENCH_FILL(A->data, n);
  BENCH_FILL(B->data, n);

  Eigen::Map<EigenMatrix> eA(A->data, 1, n);
  Eigen::Map<EigenMatrix> eB(B->data, 1, n);
  EigenMatrix eC(1, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_hadamard(C, A, B);
    eC.noalias() = eA.cwiseProduct(eB);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_hadamard(C, A, B);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      eC.noalias() = eA.cwiseProduct(eB);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // read 2n + write n = 3n
  double gb_libmat = (3.0 * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_eigen = (3.0 * n * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(A); mat_free_mat(B); mat_free_mat(C);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== HADAMARD BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("C = A .* B (element-wise)\n");

  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
