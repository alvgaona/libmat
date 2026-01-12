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
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

#ifdef MAT_HAS_OPENMP
  #define OMP_STATUS "ENABLED"
  #define OMP_THREADS omp_get_max_threads()
#else
  #define OMP_STATUS "DISABLED"
  #define OMP_THREADS 1
#endif

void bench_speed(size_t m, size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", m, n);

  // libmat
  Mat *A = mat_mat(m, n);
  Mat *At = mat_mat(n, m);
  BENCH_FILL(A->data, m * n);

  // Eigen
  Eigen::Map<EigenMatrix> eA(A->data, m, n);
  EigenMatrix eAt(n, m);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_t(At, A);
    eAt.noalias() = eA.transpose();
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < iterations; i++)
      mat_t(At, A);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / iterations / 1000.0;

    start = bench_now();
    for (int i = 0; i < iterations; i++)
      eAt.noalias() = eA.transpose();
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / iterations / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  double gb_libmat = (2.0 * m * n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_eigen = (2.0 * m * n * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(A); mat_free_mat(At);
}

int main() {
  srand(42);
  bench_init();

  printf("=== TRANSPOSE BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("B = A^T\n");
  printf("libmat OpenMP: %s", OMP_STATUS);
#ifdef MAT_HAS_OPENMP
  printf(" (threads: %d, threshold: %d)", OMP_THREADS, MAT_OMP_THRESHOLD);
#endif
  printf("\n");
  printf("Eigen threads: %d\n", Eigen::nbThreads());

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 1000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);
  bench_speed(2048, 2048, 20);
  bench_speed(4096, 4096, 5);

  return 0;
}
