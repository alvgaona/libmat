CC = gcc
CFLAGS = -Wall -Wextra -Iinclude

all: build/libmat.a build/main

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libmat.a: build/mat.o build/util.o
	ar rcs $@ $^

build/main: src/main.c build/libmat.a
	$(CC) $(CFLAGS) $< -Lbuild -lmat -o $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
