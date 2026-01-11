# libmat

An stb single-file linear algebra library in pure C

## Build

```bash
make          # build examples and tests
make check    # run tests
make clean    # cleanup
```

## Benchmarks

### Scalar vs NEON

Run `make bench` to compare scalar and NEON implementations:

```bash
make bench (run standard benches)
```

### vs Eigen

- Benchmarks against Eigen are in `tests/bench/eigen/` 
- Requires Eigen installed

```bash
# Example: build and run GEMM benchmark
c++ -O3 -I. -I/opt/homebrew/include/eigen3 \
  tests/bench/eigen/bench_gemm.cpp -o bench_gemm
./bench_gemm
```

### vs OpenBLAS

- Benchmarks against OpenBLAS are in `tests/bench/openblas/`
- Requires OpenBLAS installed.

```bash
# Example: build and run GEMM benchmark
cc -O3 -I. -I/opt/homebrew/opt/openblas/include \
  -L/opt/homebrew/opt/openblas/lib \
  tests/bench/openblas/bench_gemm_blas.c -o bench_gemm_blas -lopenblas -lm
./bench_gemm_blas
```
