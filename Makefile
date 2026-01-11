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

.PHONY: all examples test check bench clean

all: examples test

examples: $(EXAMPLES) $(EXAMPLES_CXX)

test: $(TESTS)

examples/%: examples/%.c mat.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

examples/%: examples/%.cpp mat.h
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

tests/%: tests/%.c mat.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

check: test
	@for t in $(TESTS); do echo "Running $$t..."; ./$$t || exit 1; done
	@echo "All tests passed!"

tests/bench/bench_all: tests/bench/bench_all.c mat.h
	$(CC) $(CFLAGS) -O3 -o $@ $< $(LDLIBS)

bench: tests/bench/bench_all
	./tests/bench/bench_all

clean:
	find examples tests -type f ! -name "*.c" ! -name "*.cpp" ! -name "*.h" -delete
	rm -rf examples/*.dSYM tests/*.dSYM
