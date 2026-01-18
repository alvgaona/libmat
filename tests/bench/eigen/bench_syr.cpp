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

void bench_syr_lower(size_t n, int iterations) {
  printf("\n--- %zux%zu (lower) ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Vec *x = mat_vec(n);
  BENCH_FILL(A->data, n * n);
  BENCH_FILL(x->data, n);

  EigenMatrix eA(n, n);
  eA = Eigen::Map<EigenMatrix>(A->data, n, n);
  Eigen::Map<EigenVector> ex(x->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syr(A, alpha, x, 'L');
    eA.selfadjointView<Eigen::Lower>().rankUpdate(ex, alpha);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_syr(A, alpha, x, 'L');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eA.selfadjointView<Eigen::Lower>().rankUpdate(ex, alpha);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPS for SYR lower: n*(n+1)/2 multiply-adds = n*(n+1) flops
  double flops = (double)n * (n + 1);
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_eigen = flops / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A); mat_free_mat(x);
}

void bench_syr_upper(size_t n, int iterations) {
  printf("\n--- %zux%zu (upper) ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Vec *x = mat_vec(n);
  BENCH_FILL(A->data, n * n);
  BENCH_FILL(x->data, n);

  EigenMatrix eA(n, n);
  eA = Eigen::Map<EigenMatrix>(A->data, n, n);
  Eigen::Map<EigenVector> ex(x->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_syr(A, alpha, x, 'U');
    eA.selfadjointView<Eigen::Upper>().rankUpdate(ex, alpha);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_syr(A, alpha, x, 'U');
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eA.selfadjointView<Eigen::Upper>().rankUpdate(ex, alpha);
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPS for SYR upper: n*(n+1)/2 multiply-adds = n*(n+1) flops
  double flops = (double)n * (n + 1);
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_eigen = flops / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A); mat_free_mat(x);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== SYR BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A += alpha * x * x^T (symmetric rank-1 update)\n");

  printf("\n========== Lower Triangle ==========\n");
  bench_syr_lower(64, 10000);
  bench_syr_lower(128, 5000);
  bench_syr_lower(256, 1000);
  bench_syr_lower(512, 500);
  bench_syr_lower(1024, 100);

  printf("\n========== Upper Triangle ==========\n");
  bench_syr_upper(64, 10000);
  bench_syr_upper(128, 5000);
  bench_syr_upper(256, 1000);
  bench_syr_upper(512, 500);
  bench_syr_upper(1024, 100);

  return 0;
}
