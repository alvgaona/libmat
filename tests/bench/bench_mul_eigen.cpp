#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "../../mat.h"

#include <Eigen/Dense>

Mat *mat_random(size_t rows, size_t cols) {
  Mat *m = mat_mat(rows, cols);
  for (size_t i = 0; i < rows * cols; i++) {
    m->data[i] = (float)rand() / RAND_MAX * 200.0f - 100.0f;
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

double bench_libmat(Mat *out, const Mat *a, const Mat *b, int iters) {
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
      mat_mul(out, a, b);
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / iters;
    if (t < best) best = t;
  }
  return best;
}

double bench_eigen(int n, int iters) {
  Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n);
  Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
  Eigen::MatrixXf C(n, n);

  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
      C.noalias() = A * B;
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / iters;
    if (t < best) best = t;
  }
  return best;
}

int main() {
  init_timer();

  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};

  printf("## mat_mul vs Eigen\n\n");
  printf("| Size | libmat | Eigen | Ratio |\n");
  printf("|------|--------|-------|-------|\n");

  for (int i = 0; i < 9; i++) {
    int n = sizes[i];
    Mat *a = mat_random(n, n);
    Mat *b = mat_random(n, n);
    Mat *out = mat_mat(n, n);

    int iters = (n <= 64) ? 100 : (n <= 256) ? 10 : 3;

    double libmat = bench_libmat(out, a, b, iters);
    double eigen = bench_eigen(n, iters);

    if (libmat < 1000) {
      printf("| %dx%d | %.0f ns | %.0f ns | %.2fx |\n", n, n, libmat, eigen, libmat/eigen);
    } else if (libmat < 1000000) {
      printf("| %dx%d | %.1f us | %.1f us | %.2fx |\n", n, n, libmat/1000, eigen/1000, libmat/eigen);
    } else {
      printf("| %dx%d | %.1f ms | %.1f ms | %.2fx |\n", n, n, libmat/1000000, eigen/1000000, libmat/eigen);
    }

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
  }

  return 0;
}
