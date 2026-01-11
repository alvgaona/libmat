# libmat

An stb single-file linear algebra library in pure C

## Build

```bash
make          # build examples and tests
make check    # run tests
make clean    # cleanup
```

## IDE Support

Generate `compile_flags.txt` for clangd:

```bash
./generate_compile_flags.sh > compile_flags.txt
```

This auto-detects Eigen and OpenBLAS via `pkg-config` (Linux) or Homebrew paths (macOS).

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
- **Important:** Homebrew's OpenBLAS ignores thread settings, giving unfair multi-threaded results. Build single-threaded OpenBLAS for accurate comparison:

```bash
# Build single-threaded OpenBLAS with LAPACK
git clone --depth 1 https://github.com/OpenMathLib/OpenBLAS.git /tmp/OpenBLAS
cd /tmp/OpenBLAS
make -j8 libs USE_OPENMP=0 USE_THREAD=0 NO_SHARED=1
make PREFIX=/path/to/libmat/deps/openblas USE_OPENMP=0 USE_THREAD=0 NO_SHARED=1 install
```

Build flags:
- `USE_THREAD=0` - Disable threading (critical for fair single-core benchmarks)
- `USE_OPENMP=0` - Disable OpenMP
- `NO_SHARED=1` - Static library only
- `libs` target - Skip tests (avoids gfortran linking issues)

```bash
# Run benchmark
cd tests/bench/openblas
cc -O3 -I../../.. -I.. -I../../../deps/openblas/include -L../../../deps/openblas/lib bench_gemm_blas.c -o bench_gemm -lopenblas && ./bench_gemm
```
