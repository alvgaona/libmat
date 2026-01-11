# libmat

An stb single-file linear algebra library in pure C

## Build

```bash
make          # build examples and tests
make check    # run tests
make bench    # run benchmarks
make clean    # cleanup
```

## Benchmarks

Benchmarks compare against OpenBLAS. To build:

```bash
clang -O3 -o bench tests/bench/bench_norm2_blas.c \
  -I/path/to/openblas/include -L/path/to/openblas/lib -lopenblas -lm
```
