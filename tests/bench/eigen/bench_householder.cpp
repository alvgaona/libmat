#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 100
#define BENCH_ROUNDS 10
#define BENCH_WARMUP 3
#define BENCH_IMPLEMENTATION
#include "bench.h"

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenMatrix = Eigen::MatrixXd;
  using EigenVector = Eigen::VectorXd;
  using Scalar = double;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  using Scalar = float;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

void bench_householder_compute(size_t n) {
  printf("\n--- Compute Householder (n=%zu) ---\n", n);

  Vec *x = mat_vec(n);
  Vec *v = mat_vec(n);
  BENCH_FILL(x->data, n);

  EigenVector ex(n);
  for (size_t i = 0; i < n; i++) ex(i) = x->data[i];

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_elem_t tau;
    mat_householder(v, &tau, x);

    EigenVector ev(n);
    Scalar etau, ebeta;
    ex.makeHouseholder(ev, etau, ebeta);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      mat_elem_t tau;
      mat_householder(v, &tau, x);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      EigenVector ev(n);
      Scalar etau, ebeta;
      ex.makeHouseholder(ev, etau, ebeta);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f ns  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f ns\n", es.avg, es.std);

  mat_free_mat(x);
  mat_free_mat(v);
}

void bench_householder_apply_left(size_t m, size_t n) {
  printf("\n--- Apply Householder Left (%zux%zu) ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(m);
  Vec *v = mat_vec(m);
  BENCH_FILL(A->data, m * n);
  BENCH_FILL(x->data, m);

  mat_elem_t tau;
  mat_householder(v, &tau, x);

  EigenMatrix eA(m, n);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      eA(i, j) = A->data[i * n + j];

  EigenVector ev(m);
  Scalar etau, ebeta;
  Eigen::Map<EigenVector>(x->data, m).makeHouseholder(ev, etau, ebeta);

  Scalar *workspace = new Scalar[n];

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_householder_left(A, v, tau);
    eA.applyHouseholderOnTheLeft(ev, etau, workspace);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      mat_householder_left(A, v, tau);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      eA.applyHouseholderOnTheLeft(ev, etau, workspace);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;
  }

  delete[] workspace;

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f ns  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f ns\n", es.avg, es.std);

  mat_free_mat(A);
  mat_free_mat(x);
  mat_free_mat(v);
}

void bench_householder_apply_right(size_t m, size_t n) {
  printf("\n--- Apply Householder Right (%zux%zu) ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(n);
  Vec *v = mat_vec(n);
  BENCH_FILL(A->data, m * n);
  BENCH_FILL(x->data, n);

  mat_elem_t tau;
  mat_householder(v, &tau, x);

  EigenMatrix eA(m, n);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      eA(i, j) = A->data[i * n + j];

  EigenVector ev(n);
  Scalar etau, ebeta;
  Eigen::Map<EigenVector>(x->data, n).makeHouseholder(ev, etau, ebeta);

  Scalar *workspace = new Scalar[m];

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_householder_right(A, v, tau);
    eA.applyHouseholderOnTheRight(ev, etau, workspace);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      mat_householder_right(A, v, tau);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      eA.applyHouseholderOnTheRight(ev, etau, workspace);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS;
  }

  delete[] workspace;

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f ns  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f ns\n", es.avg, es.std);

  mat_free_mat(A);
  mat_free_mat(x);
  mat_free_mat(v);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== HOUSEHOLDER BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  printf("\n== Computing Householder vector ==\n");
  bench_householder_compute(32);
  bench_householder_compute(64);
  bench_householder_compute(128);
  bench_householder_compute(256);

  printf("\n== Applying Householder from Left ==\n");
  bench_householder_apply_left(64, 64);
  bench_householder_apply_left(128, 128);
  bench_householder_apply_left(256, 256);

  printf("\n== Applying Householder from Right ==\n");
  bench_householder_apply_right(64, 64);
  bench_householder_apply_right(128, 128);
  bench_householder_apply_right(256, 256);

  return 0;
}
