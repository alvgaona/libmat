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

void bench_speed(size_t m, size_t n) {
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Mat *Q = mat_mat(m, m);
  Mat *R = mat_mat(m, n);
  BENCH_FILL(A->data, m * n);

  Eigen::Map<EigenMatrix> eA(A->data, m, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_qr(A, Q, R);
    Eigen::HouseholderQR<EigenMatrix> qr(eA);
    (void)qr.householderQ();
    (void)qr.matrixQR();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_qr(A, Q, R);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      Eigen::HouseholderQR<EigenMatrix> qr(eA);
      EigenMatrix eQ = qr.householderQ();
      EigenMatrix eR = qr.matrixQR().triangularView<Eigen::Upper>();
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f us\n", es.avg, es.std);

  mat_free_mat(A); mat_free_mat(Q); mat_free_mat(R);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== QR BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A = Q * R (Householder)\n");

  bench_speed(32, 32);
  bench_speed(64, 64);
  bench_speed(128, 128);
  bench_speed(256, 256);
  bench_speed(512, 512);

  return 0;
}
