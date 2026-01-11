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
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_speed(size_t m, size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(n);
  Vec *y = mat_vec(m);
  BENCH_FILL(A->data, m * n);
  BENCH_FILL(x->data, n);
  BENCH_FILL(y->data, m);

  Eigen::Map<EigenMatrix> eA(A->data, m, n);
  Eigen::Map<EigenVector> ex(x->data, n);
  EigenVector ey(m);

  mat_elem_t alpha = 1.0, beta = 0.0;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_gemv(y, alpha, A, x, beta);
    ey.noalias() = eA * ex;
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_gemv(y, alpha, A, x, beta);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      ey.noalias() = eA * ex;
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPs: 2*m*n (multiply-add for each element)
  double gflops_libmat = (2.0 * m * n) / (ls.avg * 1000.0);
  double gflops_eigen = (2.0 * m * n) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A); mat_free_mat(x); mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== GEMV BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("y = A * x\n");

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 2000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);
  bench_speed(2048, 2048, 20);

  return 0;
}
