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
  #define PRECISION_NAME "float64"
#else
  using EigenMatrix = Eigen::MatrixXf;
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

  // libmat
  Mat *A = mat_mat(m, n);
  Mat *At = mat_mat(n, m);
  fill_random(A->data, m * n);

  // Eigen
  Eigen::Map<EigenMatrix> eA(A->data, m, n);
  EigenMatrix eAt(n, m);

  // Warmup
  for (int i = 0; i < WARMUP; i++) {
    mat_t(At, A);
    eAt.noalias() = eA.transpose();
  }

  double libmat_times[ROUNDS], eigen_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iterations; i++)
      mat_t(At, A);
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / iterations / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < iterations; i++)
      eAt.noalias() = eA.transpose();
    end = mach_absolute_time();
    eigen_times[r] = (end - start) * ns_per_tick / iterations / 1000.0;
  }

  Stats ls = compute_stats(libmat_times, ROUNDS);
  Stats es = compute_stats(eigen_times, ROUNDS);

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
  init_timer();

  Eigen::setNbThreads(1);

  printf("=== TRANSPOSE BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("B = A^T\n");

  bench_speed(64, 64, 10000);
  bench_speed(128, 128, 5000);
  bench_speed(256, 256, 1000);
  bench_speed(512, 512, 500);
  bench_speed(1024, 1024, 100);
  bench_speed(2048, 2048, 20);

  return 0;
}
