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
cd tests/bench/eigen
c++ -O3 -I../../.. -I.. -I. -I/path/to/include/eigen3 bench_plu.cpp -o bench_plu && ./bench_plu
```

### vs OpenBLAS

- Benchmarks against OpenBLAS are in `tests/bench/openblas/`
- Requires OpenBLAS installed.

```bash
# Example: build and run GEMM benchmark
cd tests/bench/openblas
cc -O3 -I../../.. -I.. -I. -I/path/to/openblas/include -L/path/to/openblas/lib bench_axpy_blas.c -o bench_axpy -lopenblas -lm && ./bench_axpy
```
