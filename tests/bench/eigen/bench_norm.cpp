#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <mach/mach_time.h>
#include <Eigen/Dense>

#define MAT_IMPLEMENTATION
#include "mat.h"

#ifdef MAT_DOUBLE_PRECISION
  using EigenVector = Eigen::VectorXd;
  #define PRECISION_NAME "float64"
#else
  using EigenVector = Eigen::VectorXf;
  #define PRECISION_NAME "float32"
#endif

#define ITERATIONS 10000
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

volatile mat_elem_t sink;

void bench_speed(size_t n) {
  printf("\n--- Size: %zu ---\n", n);

  Vec *x = mat_vec(n);
  fill_random(x->data, n);

  Eigen::Map<EigenVector> ex(x->data, n);

  for (int i = 0; i < WARMUP; i++) {
    sink = mat_norm_fro(x);
    sink = mat_norm_fro_fast(x);
    sink = ex.norm();
  }

  double libmat_times[ROUNDS], libmat_fast_times[ROUNDS], eigen_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      sink = mat_norm_fro(x);
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      sink = mat_norm_fro_fast(x);
    end = mach_absolute_time();
    libmat_fast_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      sink = ex.norm();
    end = mach_absolute_time();
    eigen_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  Stats ls = compute_stats(libmat_times, ROUNDS);
  Stats lf = compute_stats(libmat_fast_times, ROUNDS);
  Stats es = compute_stats(eigen_times, ROUNDS);

  double gb_libmat = (n * sizeof(mat_elem_t)) / (ls.avg * 1000.0);
  double gb_fast = (n * sizeof(mat_elem_t)) / (lf.avg * 1000.0);
  double gb_eigen = (n * sizeof(mat_elem_t)) / (es.avg * 1000.0);

  printf("libmat safe: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         ls.avg, ls.std, es.avg / ls.avg, gb_libmat);
  printf("libmat fast: %8.2f ± %.2f us  (%.1fx vs Eigen)  %.1f GB/s\n",
         lf.avg, lf.std, es.avg / lf.avg, gb_fast);
  printf("Eigen:       %8.2f ± %.2f us                    %.1f GB/s\n",
         es.avg, es.std, gb_eigen);

  mat_free_mat(x);
}

int main() {
  srand(42);
  init_timer();
  Eigen::setNbThreads(1);

  printf("=== NORM BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("result = ||x||_2\n");

  bench_speed(100);
  bench_speed(1000);
  bench_speed(10000);
  bench_speed(100000);
  bench_speed(1000000);

  return 0;
}
