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
  using EigenVec = Eigen::VectorXd;
  using EigenMat = Eigen::MatrixXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenVec = Eigen::VectorXf;
  using EigenMat = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

volatile float sink;

void bench_speed(size_t m, size_t n) {
  printf("\n--- Size: %zu x %zu ---\n", m, n);

  Vec *v1 = mat_vec(m);
  Vec *v2 = mat_vec(n);
  Mat *out = mat_mat(m, n);
  BENCH_FILL(v1->data, m);
  BENCH_FILL(v2->data, n);

  Eigen::Map<EigenVec> ev1(v1->data, m);
  Eigen::Map<EigenVec> ev2(v2->data, n);
  EigenMat eOut(m, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_outer(out, v1, v2);
    eOut.noalias() = ev1 * ev2.transpose();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      mat_outer(out, v1, v2);
      sink = out->data[0];
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      eOut.noalias() = ev1 * ev2.transpose();
      sink = eOut(0, 0);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(v1);
  mat_free_mat(v2);
  mat_free_mat(out);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== OUTER PRODUCT BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  bench_speed(100, 100);
  bench_speed(256, 256);
  bench_speed(512, 512);
  bench_speed(1000, 1000);

  return 0;
}
