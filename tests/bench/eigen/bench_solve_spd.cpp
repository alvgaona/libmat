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
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
  #define BENCH_FILL bench_fill_random_d
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
  #define BENCH_FILL bench_fill_random_f
#endif

// Create SPD matrix: A = M * M^T + epsilon * I
void make_spd(mat_elem_t *A, size_t n) {
  mat_elem_t *M = (mat_elem_t *)malloc(n * n * sizeof(mat_elem_t));
  BENCH_FILL(M, n * n);

  // A = M * M^T
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < n; k++) {
        sum += M[i * n + k] * M[j * n + k];
      }
      A[i * n + j] = sum;
    }
  }
  // Add epsilon * I
  for (size_t i = 0; i < n; i++) {
    A[i * n + i] += 0.1f;
  }
  free(M);
}

void bench_speed(size_t n) {
  printf("\n--- %zux%zu ---\n", n, n);

  // Allocate libmat matrices
  Mat *A = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  // Create SPD matrix
  make_spd(A->data, n);
  BENCH_FILL(b->data, n);

  // Map to Eigen (row-major)
  Eigen::Map<EigenMatrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
    eA(A->data, n, n, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
  Eigen::Map<EigenVector> eb(b->data, n);
  EigenVector ex(n);

  // Warmup
  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_spd(x, A, b);
    ex = eA.llt().solve(eb);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_solve_spd(x, A, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      ex = eA.llt().solve(eb);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f us\n", es.avg, es.std);

  mat_free_mat(A);
  mat_free_mat(b);
  mat_free_mat(x);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== SOLVE SPD BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("Solve Ax = b for SPD A (Cholesky: LLT)\n");

  bench_speed(32);
  bench_speed(64);
  bench_speed(128);
  bench_speed(256);
  bench_speed(512);

  return 0;
}
