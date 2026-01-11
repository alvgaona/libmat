#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ROUNDS 20
#define BENCH_WARMUP 10
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenMatrix = Eigen::MatrixXd;
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
#endif

void bench_speed(size_t m, size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(m);
  Vec *y = mat_vec(n);
  bench_fill_random_f(A->data, m * n);
  bench_fill_random_f(x->data, m);
  bench_fill_random_f(y->data, n);

  EigenMatrix eA(m, n);
  eA = Eigen::Map<EigenMatrix>(A->data, m, n);
  Eigen::Map<EigenVector> ex(x->data, m);
  Eigen::Map<EigenVector> ey(y->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_ger(A, alpha, x, y);
    eA.noalias() += alpha * ex * ey.transpose();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_ger(A, alpha, x, y);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eA.noalias() += alpha * ex * ey.transpose();
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // Bandwidth: read m + read n + read/write m*n
  double gb_libmat = ((m + n + 2*m*n) * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_eigen = ((m + n + 2*m*n) * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(A); mat_free_mat(x); mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== GER BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A += alpha * x * y^T\n");

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 1000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);

  return 0;
}
