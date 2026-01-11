#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mach/mach_time.h>

#define MAT_EXPOSE_INTERNALS
#define MAT_IMPLEMENTATION
#include "../../mat.h"

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

typedef void (*gemm_fn)(Mat*, mat_elem_t, const Mat*, const Mat*, mat_elem_t);

double bench_gemm(gemm_fn fn, Mat *out, const Mat *a, const Mat *b, int iters) {
  double best = DBL_MAX;
  for (int r = 0; r < ROUNDS; r++) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
      fn(out, 1, a, b, 0);
    }
    uint64_t end = mach_absolute_time();
    double t = (end - start) * ns_per_tick / iters;
    if (t < best) best = t;
  }
  return best;
}

int main() {
  init_timer();

  int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

  printf("## mat_gemm (C = alpha*A*B + beta*C)\n\n| Size | Scalar | NEON | Speedup |\n|------|--------|------|--------|\n");

  for (int i = 0; i < 10; i++) {
    int n = sizes[i];
    Mat *a = mat_random(n, n);
    Mat *b = mat_random(n, n);
    Mat *out = mat_mat(n, n);

    // Fewer iterations for large matrices
    int iters = (n <= 64) ? 100 : (n <= 256) ? 10 : 3;

    double scalar = bench_gemm(mat_gemm_scalar_impl, out, a, b, iters);
#ifdef __ARM_NEON
    double neon = bench_gemm(mat_gemm_neon_impl, out, a, b, iters);
    printf("| %dx%d | %.0f us | %.0f us | %.2fx |\n", n, n, scalar/1000, neon/1000, scalar/neon);
#else
    printf("| %dx%d | %.0f us | N/A | N/A |\n", n, n, scalar/1000);
#endif

    mat_free_mat(a);
    mat_free_mat(b);
    mat_free_mat(out);
  }

  return 0;
}
