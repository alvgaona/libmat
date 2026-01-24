CC = cc
CXX = c++
CFLAGS = -Wall -Wextra -I.
CXXFLAGS = -Wall -Wextra -I.
LDLIBS = -lm

# Examples
EXAMPLES = $(patsubst %.c,%,$(wildcard examples/*.c))
EXAMPLES_CXX = $(patsubst %.cpp,%,$(wildcard examples/*.cpp))

# Tests
TESTS = $(patsubst %.c,%,$(wildcard tests/*.c))

# ZAP benchmarks
ZAP_BENCH_DIR = tests/bench/zap
ZAP_BENCHES = $(ZAP_BENCH_DIR)/bench_zap_blas1 \
              $(ZAP_BENCH_DIR)/bench_zap_blas2 \
              $(ZAP_BENCH_DIR)/bench_zap_blas3 \
              $(ZAP_BENCH_DIR)/bench_zap_reductions \
              $(ZAP_BENCH_DIR)/bench_zap_decomp \
              $(ZAP_BENCH_DIR)/bench_zap_solvers \
              $(ZAP_BENCH_DIR)/bench_zap_matrix_ops \
              $(ZAP_BENCH_DIR)/bench_zap_advanced \
              $(ZAP_BENCH_DIR)/bench_zap_trsv \
              $(ZAP_BENCH_DIR)/bench_zap_elementwise \
              $(ZAP_BENCH_DIR)/bench_zap_stats \
              $(ZAP_BENCH_DIR)/bench_zap_norms \
              $(ZAP_BENCH_DIR)/bench_zap_misc

.PHONY: all examples test check bench clean \
        bench-zap bench-zap-blas1 bench-zap-blas2 bench-zap-blas3 \
        bench-zap-reductions bench-zap-decomp bench-zap-solvers \
        bench-zap-matrix-ops bench-zap-advanced bench-zap-trsv \
        bench-zap-elementwise bench-zap-stats bench-zap-norms bench-zap-misc

all: examples test

examples: $(EXAMPLES) $(EXAMPLES_CXX)

test: $(TESTS)

examples/%: examples/%.c mat.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

examples/%: examples/%.cpp mat.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# Exclude ZAP benchmarks from generic test rule (they need -O3)
tests/%: tests/%.c mat.h
	@if echo "$@" | grep -q "bench/zap"; then \
		$(CC) $(CFLAGS) -O3 -o $@ $< $(LDLIBS); \
	else \
		$(CC) $(CFLAGS) -o $@ $< $(LDLIBS); \
	fi

check: test
	@for t in $(TESTS); do echo "Running $$t..."; ./$$t || exit 1; done
	@echo "All tests passed!"

tests/bench/bench_all: tests/bench/bench_all.c mat.h
	$(CC) $(CFLAGS) -O3 -o $@ $< $(LDLIBS)

bench: tests/bench/bench_all
	./tests/bench/bench_all

# ZAP benchmark build rules
$(ZAP_BENCH_DIR)/%: $(ZAP_BENCH_DIR)/%.c $(ZAP_BENCH_DIR)/zap.h mat.h
	$(CC) $(CFLAGS) -O3 -o $@ $< $(LDLIBS)

# ZAP benchmark targets
bench-zap: $(ZAP_BENCHES)
	@for b in $(ZAP_BENCHES); do echo "Running $$b..."; ./$$b || exit 1; done

bench-zap-blas1: $(ZAP_BENCH_DIR)/bench_zap_blas1
	./$(ZAP_BENCH_DIR)/bench_zap_blas1

bench-zap-blas2: $(ZAP_BENCH_DIR)/bench_zap_blas2
	./$(ZAP_BENCH_DIR)/bench_zap_blas2

bench-zap-blas3: $(ZAP_BENCH_DIR)/bench_zap_blas3
	./$(ZAP_BENCH_DIR)/bench_zap_blas3

bench-zap-reductions: $(ZAP_BENCH_DIR)/bench_zap_reductions
	./$(ZAP_BENCH_DIR)/bench_zap_reductions

bench-zap-decomp: $(ZAP_BENCH_DIR)/bench_zap_decomp
	./$(ZAP_BENCH_DIR)/bench_zap_decomp

bench-zap-solvers: $(ZAP_BENCH_DIR)/bench_zap_solvers
	./$(ZAP_BENCH_DIR)/bench_zap_solvers

bench-zap-matrix-ops: $(ZAP_BENCH_DIR)/bench_zap_matrix_ops
	./$(ZAP_BENCH_DIR)/bench_zap_matrix_ops

bench-zap-advanced: $(ZAP_BENCH_DIR)/bench_zap_advanced
	./$(ZAP_BENCH_DIR)/bench_zap_advanced

bench-zap-trsv: $(ZAP_BENCH_DIR)/bench_zap_trsv
	./$(ZAP_BENCH_DIR)/bench_zap_trsv

bench-zap-elementwise: $(ZAP_BENCH_DIR)/bench_zap_elementwise
	./$(ZAP_BENCH_DIR)/bench_zap_elementwise

bench-zap-stats: $(ZAP_BENCH_DIR)/bench_zap_stats
	./$(ZAP_BENCH_DIR)/bench_zap_stats

bench-zap-norms: $(ZAP_BENCH_DIR)/bench_zap_norms
	./$(ZAP_BENCH_DIR)/bench_zap_norms

bench-zap-misc: $(ZAP_BENCH_DIR)/bench_zap_misc
	./$(ZAP_BENCH_DIR)/bench_zap_misc

clean:
	find examples tests -type f ! -name "*.c" ! -name "*.cpp" ! -name "*.h" -delete
	rm -rf examples/*.dSYM tests/*.dSYM
