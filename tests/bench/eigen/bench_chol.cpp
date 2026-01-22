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

// Make matrix SPD: A = B * B^T + n*I
void make_spd(Mat *A) {
  size_t n = A->rows;
  Mat *B = mat_mat(n, n);
  BENCH_FILL(B->data, n * n);

  // A = B * B^T using MAT_AT/MAT_SET for storage-order independence
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= i; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += MAT_AT(B, i, k) * MAT_AT(B, j, k);
      }
      MAT_SET(A, i, j, sum);
      MAT_SET(A, j, i, sum);
    }
  }

  // Add n*I
  for (size_t i = 0; i < n; i++) {
    MAT_SET(A, i, i, MAT_AT(A, i, i) + (mat_elem_t)n);
  }

  mat_free_mat(B);
}

void bench_speed(size_t n) {
  printf("\n--- %zux%zu ---\n", n, n);

  Mat *A = mat_mat(n, n);
  Mat *L = mat_mat(n, n);

  make_spd(A);

  Eigen::Map<EigenMatrix> eA(A->data, n, n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_chol(A, L);
    Eigen::LLT<EigenMatrix> llt(eA);
    (void)llt.matrixL();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_chol(A, L);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      Eigen::LLT<EigenMatrix> llt(eA);
      (void)llt.matrixL();
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  // FLOPS for Cholesky: ~n^3/3
  double flops = (double)n * n * n / 3.0;
  double gflops_libmat = flops / (ls.avg * 1000.0);
  double gflops_eigen = flops / (es.avg * 1000.0);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)  %.1f GFLOPS\n",
         ls.avg, ls.std, es.avg / ls.avg, gflops_libmat);
  printf("Eigen:  %8.1f ± %.1f us                    %.1f GFLOPS\n",
         es.avg, es.std, gflops_eigen);

  mat_free_mat(A);
  mat_free_mat(L);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== CHOLESKY BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A = L * L^T (Cholesky decomposition)\n");

  bench_speed(32);
  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
