#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/SVD>

#define BENCH_ROUNDS 10
#define BENCH_WARMUP 3
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

void bench_svd(size_t m, size_t n, int iterations) {
  size_t k = m < n ? m : n;
  printf("\n--- %zux%zu ---\n", m, n);

  // Allocate for libmat
  Mat *A = mat_mat(m, n);
  Mat *U = mat_mat(m, m);
  Vec *S = mat_vec(k);
  Mat *Vt = mat_mat(n, n);

  // Fill with random data
  BENCH_FILL(A->data, m * n);

  // Eigen matrix (column-major by default, but we use row-major Map)
  Eigen::Map<Eigen::Matrix<mat_elem_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(A->data, m, n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_svd(A, U, S, Vt);
    Eigen::JacobiSVD<EigenMatrix> jacobiSvd(eA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    (void)jacobiSvd;
  }

  double libmat_times[BENCH_ROUNDS], jacobi_times[BENCH_ROUNDS], bdcsvd_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    // libmat (Jacobi)
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_svd(A, U, S, Vt);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    // Eigen JacobiSVD
    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      Eigen::JacobiSVD<EigenMatrix> jacobiSvd(eA, Eigen::ComputeFullU | Eigen::ComputeFullV);
      (void)jacobiSvd;
    }
    end = bench_now();
    jacobi_times[r] = bench_ns(start, end) / iterations / 1000.0;

    // Eigen BDCSVD (divide & conquer)
    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      Eigen::BDCSVD<EigenMatrix> bdcSvd(eA, Eigen::ComputeFullU | Eigen::ComputeFullV);
      (void)bdcSvd;
    }
    end = bench_now();
    bdcsvd_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats js = bench_stats(jacobi_times, BENCH_ROUNDS);
  BenchStats bs = bench_stats(bdcsvd_times, BENCH_ROUNDS);

  printf("libmat (Jacobi):  %10.1f ± %5.1f us\n", ls.avg, ls.std);
  printf("Eigen JacobiSVD:  %10.1f ± %5.1f us  (%.1fx)\n", js.avg, js.std, ls.avg / js.avg);
  printf("Eigen BDCSVD:     %10.1f ± %5.1f us  (%.1fx)\n", bs.avg, bs.std, ls.avg / bs.avg);

  mat_free_mat(A);
  mat_free_mat(U);
  mat_free_mat(S);
  mat_free_mat(Vt);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== SVD BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("libmat: one-sided Jacobi\n");
  printf("JacobiSVD: Eigen's two-sided Jacobi\n");
  printf("BDCSVD: bidiag + divide & conquer\n");
  printf("Rounds: %d\n", BENCH_ROUNDS);

  bench_svd(10, 10, 1000);
  bench_svd(20, 20, 500);
  bench_svd(50, 50, 100);
  bench_svd(100, 100, 20);
  bench_svd(200, 200, 5);
  bench_svd(50, 30, 100);   // tall
  bench_svd(30, 50, 100);   // wide

  return 0;
}
