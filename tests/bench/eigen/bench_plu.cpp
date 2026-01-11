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
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
#endif

void bench_speed(size_t n) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *L = mat_mat(n, n);
  Mat *U = mat_mat(n, n);
  Perm *p = mat_perm(n);

  bench_fill_random_f(A->data, n * n);

  Eigen::Map<EigenMatrix> eA(A->data, n, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_plu(A, L, U, p);
    Eigen::PartialPivLU<EigenMatrix> lu(eA);
    (void)lu.matrixLU();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_plu(A, L, U, p);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      Eigen::PartialPivLU<EigenMatrix> lu(eA);
      (void)lu.matrixLU();
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
  mat_free_mat(L);
  mat_free_mat(U);
  mat_free_perm(p);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== PLU BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("P * A = L * U (Partial Pivoting)\n");

  bench_speed(32);
  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
