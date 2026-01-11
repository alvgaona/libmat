#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "../../mat.h"

#ifdef BENCH_EIGEN
#include <Eigen/Dense>
#endif

Mat *mat_random(size_t rows, size_t cols) {
  Mat *m = mat_mat(rows, cols);
  for (size_t i = 0; i < rows * cols; i++) {
    m->data[i] = (float)rand() / RAND_MAX * 200.0f - 100.0f;  // [-100, 100]
  }
  return m;
}

#define ROUNDS 3

static double ns_per_tick;

void init_timer(void) {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  ns_per_tick = (double)info.numer / info.denom;
}

typedef void (*qr_fn)(const Mat*, Mat*, Mat*);

double bench_qr(qr_fn fn, const Mat *A, Mat *Q, Mat *R, int iters) {
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
      fn(A, Q, R);
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / iters;
    if (t < best) best = t;
  }
  return best;
}

#ifdef BENCH_EIGEN
double bench_eigen_qr(int n, int iters) {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n);
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
      Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
      Eigen::MatrixXf Q = qr.householderQ();
      Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();
      (void)Q; (void)R;
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / iters;
    if (t < best) best = t;
  }
  return best;
}
#endif

int main() {
  init_timer();

  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512};

#ifdef BENCH_EIGEN
  printf("## mat_qr vs Eigen\n\n");
  printf("| Size | libmat | Eigen | Ratio |\n");
  printf("|------|--------|-------|-------|\n");

  for (int i = 0; i < 8; i++) {
    int n = sizes[i];
    Mat *A = mat_random(n, n);
    Mat *Q = mat_mat(n, n);
    Mat *R = mat_mat(n, n);

    int iters = (n <= 64) ? 100 : (n <= 256) ? 10 : 3;

    double libmat = bench_qr(mat_qr_neon_impl, A, Q, R, iters);
    double eigen = bench_eigen_qr(n, iters);

    if (libmat < 1000) {
      printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", n, n, libmat, eigen, libmat/eigen);
    } else if (libmat < 1000000) {
      printf("| %dx%d | %.1f us | %.1f us | %.2fx |\n", n, n, libmat/1000, eigen/1000, libmat/eigen);
    } else {
      printf("| %dx%d | %.1f ms | %.1f ms | %.2fx |\n", n, n, libmat/1000000, eigen/1000000, libmat/eigen);
    }

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
  }
#else
  printf("## mat_qr (Householder)\n\n");
  printf("| Size | Scalar | NEON | Speedup |\n");
  printf("|------|--------|------|--------|\n");

  for (int i = 0; i < 8; i++) {
    int n = sizes[i];
    Mat *A = mat_random(n, n);
    Mat *Q = mat_mat(n, n);
    Mat *R = mat_mat(n, n);

    int iters = (n <= 64) ? 100 : (n <= 256) ? 10 : 3;

    double scalar = bench_qr(mat_qr_scalar_impl, A, Q, R, iters);
#ifdef __ARM_NEON
    double neon = bench_qr(mat_qr_neon_impl, A, Q, R, iters);
    if (scalar < 1000) {
      printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", n, n, scalar, neon, scalar/neon);
    } else if (scalar < 1000000) {
      printf("| %dx%d | %.1f us | %.1f us | %.2fx |\n", n, n, scalar/1000, neon/1000, scalar/neon);
    } else {
      printf("| %dx%d | %.1f ms | %.1f ms | %.2fx |\n", n, n, scalar/1000000, neon/1000000, scalar/neon);
    }
#else
    if (scalar < 1000) {
      printf("| %dx%d | %.0f ns | N/A | N/A |\n", n, n, scalar);
    } else if (scalar < 1000000) {
      printf("| %dx%d | %.1f us | N/A | N/A |\n", n, n, scalar/1000);
    } else {
      printf("| %dx%d | %.1f ms | N/A | N/A |\n", n, n, scalar/1000000);
    }
#endif

    mat_free_mat(A);
    mat_free_mat(Q);
    mat_free_mat(R);
  }
#endif

  return 0;
}
