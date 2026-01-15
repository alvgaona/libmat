#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cstdio>
#include <cstdlib>

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

void bench_pinv(size_t m, size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", m, n);

  // Allocate for libmat
  Mat *A = mat_mat(m, n);
  Mat *Ap = mat_mat(n, m); // pinv is n x m

  // Fill with random data
  BENCH_FILL(A->data, m * n);

  // Eigen matrix (row-major to match libmat)
  Eigen::Map<Eigen::Matrix<mat_elem_t, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>>
      eA(A->data, m, n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_pinv(Ap, A);
    Eigen::JacobiSVD<EigenMatrix> svd(eA, Eigen::ComputeFullU |
                                              Eigen::ComputeFullV);
    (void)svd;
  }

  double libmat_times[BENCH_ROUNDS], jacobi_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    // libmat pinv (via Jacobi SVD)
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++) {
      mat_pinv(Ap, A);
    }
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    // Eigen JacobiSVD-based pinv
    start = bench_now();
    for (int i = 0; i < iterations; i++) {
      Eigen::JacobiSVD<EigenMatrix> svd(eA, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
      EigenMatrix eAp = svd.solve(EigenMatrix::Identity(m, m));
      (void)eAp;
    }
    end = bench_now();
    jacobi_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats js = bench_stats(jacobi_times, BENCH_ROUNDS);

  printf("libmat:      %10.1f ± %5.1f us\n", ls.avg, ls.std);
  printf("Eigen:       %10.1f ± %5.1f us  (%.2fx)\n", js.avg, js.std,
         ls.avg / js.avg);

  MAT_FREE_MAT(A);
  MAT_FREE_MAT(Ap);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== PINV BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("Both use Jacobi SVD\n");
  printf("Rounds: %d\n", BENCH_ROUNDS);

  // Square matrices
  bench_pinv(10, 10, 1000);
  bench_pinv(20, 20, 500);
  bench_pinv(50, 50, 100);
  bench_pinv(100, 100, 20);
  bench_pinv(200, 200, 5);

  // Rectangular matrices
  bench_pinv(100, 50, 50); // tall (overdetermined)
  bench_pinv(50, 100, 50); // wide (underdetermined)
  bench_pinv(200, 50, 20); // very tall
  bench_pinv(50, 200, 20); // very wide

  return 0;
}
