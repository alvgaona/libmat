CC = gcc
CFLAGS = -Wall -Wextra

all: build/libmat.a build/main

build/%.o: %.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libmat.a: build/mat.o build/util.o
	ar rcs $@ $^

build/main: main.c build/libmat.a
	$(CC) $(CFLAGS) $< -Lbuild -lmat -o $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
