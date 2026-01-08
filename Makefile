CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -I.
CXXFLAGS = -Wall -Wextra -I.

# Examples
EXAMPLES = $(patsubst %.c,%,$(wildcard examples/*.c))
EXAMPLES_CXX = $(patsubst %.cpp,%,$(wildcard examples/*.cpp))

# Tests
TESTS = $(patsubst %.c,%,$(wildcard tests/*.c))

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
	find examples tests -type f ! -name "*.c" ! -name "*.cpp" ! -name "*.h" -delete
	rm -rf examples/*.dSYM tests/*.dSYM
