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

#define ITERATIONS 1
#define ROUNDS 1
#define WARMUP 1

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

void bench_speed(size_t m, size_t n) {
  printf("\n--- %zux%zu ---\n", m, n);

  Mat *A = mat_mat(m, n);
  Mat *Q = mat_mat(m, m);
  Mat *R = mat_mat(m, n);
  fill_random(A->data, m * n);

  Eigen::Map<EigenMatrix> eA(A->data, m, n);

  for (int i = 0; i < WARMUP; i++) {
    mat_qr(A, Q, R);
    Eigen::HouseholderQR<EigenMatrix> qr(eA);
    (void)qr.householderQ();
    (void)qr.matrixQR();
  }

  double libmat_times[ROUNDS], eigen_times[ROUNDS];

  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++)
      mat_qr(A, Q, R);
    uint64_t end = mach_absolute_time();
    libmat_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;

    start = mach_absolute_time();
    for (int i = 0; i < ITERATIONS; i++) {
      Eigen::HouseholderQR<EigenMatrix> qr(eA);
      EigenMatrix eQ = qr.householderQ();
      EigenMatrix eR = qr.matrixQR().triangularView<Eigen::Upper>();
    }
    end = mach_absolute_time();
    eigen_times[r] = (end - start) * ns_per_tick / ITERATIONS / 1000.0;
  }

  Stats ls = compute_stats(libmat_times, ROUNDS);
  Stats es = compute_stats(eigen_times, ROUNDS);

  printf("libmat: %8.1f ± %.1f us  (%.2fx vs Eigen)\n",
         ls.avg, ls.std, es.avg / ls.avg);
  printf("Eigen:  %8.1f ± %.1f us\n", es.avg, es.std);

  mat_free_mat(A); mat_free_mat(Q); mat_free_mat(R);
}

int main() {
  srand(42);
  init_timer();
  Eigen::setNbThreads(1);

  printf("=== QR BENCHMARK: libmat vs Eigen [%s] ===\n", PRECISION_NAME);
  printf("A = Q * R (Householder)\n");

  bench_speed(32, 32);
  bench_speed(64, 64);
  bench_speed(128, 128);
  bench_speed(256, 256);
  bench_speed(512, 512);

  return 0;
}
