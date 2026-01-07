CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -I.
CXXFLAGS = -Wall -Wextra -I.

# Examples (feature demos)
EXAMPLES = examples/strip_mat examples/matdef_mat examples/arena_mat examples/mat_log
EXAMPLES_CXX = examples/basic_cpp

# Tests
TESTS = tests/test_mat_add tests/test_mat_sub tests/test_mat_mul \
        tests/test_mat_transpose tests/test_mat_create tests/test_mat_scale \
        tests/test_mat_hadamard tests/test_mat_dot tests/test_mat_copy \
        tests/test_mat_equals tests/test_mat_reshape tests/test_mat_diag \
        tests/test_mat_from tests/test_mat_access tests/test_mat_add_many \
        tests/test_mat_scalar

.PHONY: all examples test check clean

all: examples test

examples: $(EXAMPLES) $(EXAMPLES_CXX)

test: $(TESTS)

examples/%: examples/%.c mat.h
	$(CC) $(CFLAGS) -o $@ $<

examples/%: examples/%.cpp mat.h
	$(CXX) $(CXXFLAGS) -o $@ $<

tests/%: tests/%.c mat.h
	$(CC) $(CFLAGS) -o $@ $<

check: test
	@for t in $(TESTS); do echo "Running $$t..."; ./$$t || exit 1; done
	@echo "All tests passed!"

clean:
	rm -f $(EXAMPLES) $(EXAMPLES_CXX) $(TESTS)
