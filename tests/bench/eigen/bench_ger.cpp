#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <mach/mach_time.h>
#include <Eigen/Dense>

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenMatrix = Eigen::MatrixXd;
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
#else
  using EigenMatrix = Eigen::MatrixXf;
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
#endif

#define ROUNDS 20
#define WARMUP 10

static double ns_per_tick;

void init_timer() {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

struct Stats { double avg, min, std; };

Stats compute_stats(double *times, int n) {
  Stats s = {0, DBL_MAX, 0};
  for (int i = 0; i < n; i++) {
    s.avg += times[i];
    if (times[i] < s.min) s.min = times[i];
  }
  s.avg /= n;
  for (int i = 0; i < n; i++)
    s.std += (times[i] - s.avg) * (times[i] - s.avg);
  s.std = sqrt(s.std / n);
  return s;
}

void fill_random(mat_elem_t *data, size_t n) {
  for (size_t i = 0; i < n; i++)
    data[i] = (mat_elem_t)rand() / RAND_MAX * 2.0 - 1.0;
}

void bench_speed(size_t m, size_t n, int iterations) {
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Vec *x = mat_vec(m);
  Vec *y = mat_vec(n);
  fill_random(A->data, m * n);
  fill_random(x->data, m);
  fill_random(y->data, n);

  EigenMatrix eA(m, n);
  eA = Eigen::Map<EigenMatrix>(A->data, m, n);
  Eigen::Map<EigenVector> ex(x->data, m);
  Eigen::Map<EigenVector> ey(y->data, n);

  mat_elem_t alpha = 2.5;

  for (int i = 0; i < WARMUP; i++) {
    mat_ger(A, alpha, x, y);
    eA.noalias() += alpha * ex * ey.transpose();
  }

  double libmat_times[ROUNDS], eigen_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iterations; i++)
      mat_ger(A, alpha, x, y);
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / iterations / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < iterations; i++)
      eA.noalias() += alpha * ex * ey.transpose();
    end = mach_absolute_time();
    eigen_times[r] = (end - start) * ns_per_tick / iterations / 1000.0;
  }

  Stats ls = compute_stats(libmat_times, ROUNDS);
  Stats es = compute_stats(eigen_times, ROUNDS);

  // Bandwidth: read m + read n + read/write m*n
  double gb_libmat = ((m + n + 2*m*n) * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_eigen = ((m + n + 2*m*n) * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("Eigen:  %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(A); mat_free_mat(x); mat_free_mat(y);
}

int main() {
  srand(42);
  init_timer();
  Eigen::setNbThreads(1);

  printf("=== GER BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A += alpha * x * y^T\n");

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 1000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);

  return 0;
}
