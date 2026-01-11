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
#else
  using EigenVec = Eigen::VectorXf;
  using EigenMat = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
#endif

volatile mat_elem_t sink;

void bench_speed(size_t n) {
  printf("\n--- Size: %zu x %zu ---\n", n, n);

  Vec *x = mat_vec(n);
  Mat *A = mat_mat(n, n);
  Vec *y = mat_vec(n);
  bench_fill_random_f(x->data, n);
  bench_fill_random_f(y->data, n);
  bench_fill_random_f(A->data, n * n);

  Eigen::Map<EigenVec> ex(x->data, n);
  Eigen::Map<EigenVec> ey(y->data, n);
  Eigen::Map<EigenMat> eA(A->data, n, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    sink = mat_bilinear(x, A, y);
    sink = ex.dot(eA * ey);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    // mat_bilinear
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      sink = mat_bilinear(x, A, y);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    // Eigen
    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      sink = ex.dot(eA * ey);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(x);
  mat_free_mat(A);
  mat_free_mat(y);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== BILINEAR FORM BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
