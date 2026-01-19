#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>

#define BENCH_ITERATIONS 10
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

// Create well-conditioned lower triangular matrix
void make_tril(mat_elem_t *L, size_t n) {
  for (size_t i = 0; i < n; i++) {
    mat_elem_t row_sum = 0;
    for (size_t j = 0; j < i; j++) {
      mat_elem_t val = (mat_elem_t)(rand() % 100) / 100.0f - 0.5f;
      L[i * n + j] = val;
      row_sum += (val > 0 ? val : -val);
    }
    L[i * n + i] = row_sum + 1.0f;
    for (size_t j = i + 1; j < n; j++) {
      L[i * n + j] = 0;
    }
  }
}

void bench_tril(size_t n) {
  printf("\n--- mat_solve_tril %zux%zu ---\n", n, n);

  Mat *L = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  make_tril(L->data, n);
  BENCH_FILL(b->data, n);

  // Eigen uses column-major, our L is row-major
  // Create Eigen lower triangular view
  EigenMatrix eL(n, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      eL(i, j) = L->data[i * n + j];
    }
  }
  Eigen::Map<EigenVector> eb(b->data, n);
  EigenVector ex(n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_tril(x, L, b);
    ex = eL.triangularView<Eigen::Lower>().solve(eb);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_solve_tril(x, L, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      ex = eL.triangularView<Eigen::Lower>().solve(eb);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);
}

void bench_triu(size_t n) {
  printf("\n--- mat_solve_triu %zux%zu ---\n", n, n);

  Mat *U = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  // Create upper triangular (transpose of lower)
  make_tril(U->data, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      U->data[i * n + j] = U->data[j * n + i];
      U->data[j * n + i] = 0;
    }
  }
  BENCH_FILL(b->data, n);

  EigenMatrix eU(n, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      eU(i, j) = U->data[i * n + j];
    }
  }
  Eigen::Map<EigenVector> eb(b->data, n);
  EigenVector ex(n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_triu(x, U, b);
    ex = eU.triangularView<Eigen::Upper>().solve(eb);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_solve_triu(x, U, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      ex = eU.triangularView<Eigen::Upper>().solve(eb);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(U);
  mat_free_mat(b);
  mat_free_mat(x);
}

void bench_trilt(size_t n) {
  printf("\n--- mat_solve_trilt %zux%zu ---\n", n, n);

  Mat *L = mat_mat(n, n);
  Vec *b = mat_vec(n);
  Vec *x = mat_vec(n);

  make_tril(L->data, n);
  BENCH_FILL(b->data, n);

  EigenMatrix eL(n, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      eL(i, j) = L->data[i * n + j];
    }
  }
  Eigen::Map<EigenVector> eb(b->data, n);
  EigenVector ex(n);

  for (int i = 0; i < BENCH_WARMUP; i++) {
    mat_solve_trilt(x, L, b);
    ex = eL.triangularView<Eigen::Lower>().transpose().solve(eb);
  }

  double libmat_times[BENCH_ROUNDS], eigen_times[BENCH_ROUNDS];

  for (int r = 0; r < BENCH_ROUNDS; r++) {
    uint64_t start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++)
      mat_solve_trilt(x, L, b);
    uint64_t end = bench_now();
    libmat_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;

    start = bench_now();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
      ex = eL.triangularView<Eigen::Lower>().transpose().solve(eb);
    }
    end = bench_now();
    eigen_times[r] = bench_ns(start, end) / BENCH_ITERATIONS / 1000.0;
  }

  BenchStats ls = bench_stats(libmat_times, BENCH_ROUNDS);
  BenchStats es = bench_stats(eigen_times, BENCH_ROUNDS);

  printf("libmat: %8.2f ± %.2f us  (%.2fx vs Eigen)\n", ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.2f ± %.2f us\n", es.avg, es.std);

  mat_free_mat(L);
  mat_free_mat(b);
  mat_free_mat(x);
}

int main() {
  srand(42);
  bench_init();
  Eigen::setNbThreads(1);

  printf("=== TRSV BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);

  size_t sizes[] = {32, 64, 128, 256, 512};

  printf("\n### Lower Triangular (Lx = b)\n");
  for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
    bench_tril(sizes[i]);
  }

  printf("\n### Upper Triangular (Ux = b)\n");
  for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
    bench_triu(sizes[i]);
  }

  printf("\n### L^T solve (L^T x = b)\n");
  for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
    bench_trilt(sizes[i]);
  }

  return 0;
}
