#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Simple bump arena allocator
typedef struct {
    char *buf;
    size_t size;
    size_t offset;
} Arena;

#ifdef USE_ARENA
static Arena arena;

static void arena_init(Arena *a, size_t size) {
    a->buf = malloc(size);
    a->size = size;
    a->offset = 0;
}

static void *arena_alloc(Arena *a, size_t sz) {
    sz = (sz + 7) & ~7;  // align to 8 bytes
    if (a->offset + sz > a->size) return NULL;
    void *ptr = a->buf + a->offset;
    a->offset += sz;
    return ptr;
}

static void *arena_calloc(Arena *a, size_t n, size_t sz) {
    void *ptr = arena_alloc(a, n * sz);
    if (ptr) memset(ptr, 0, n * sz);
    return ptr;
}

static void arena_reset(Arena *a) { (void)a; a->offset = 0; }
static void arena_free(Arena *a) { free(a->buf); }
#endif

// Compile with -DUSE_ARENA to use arena allocator
#ifdef USE_ARENA
#define MAT_MALLOC(sz)    arena_alloc(&arena, sz)
#define MAT_CALLOC(n, sz) arena_calloc(&arena, n, sz)
#define MAT_FREE(p)       ((void)0)
#endif

#define MAT_IMPLEMENTATION
#include "mat.h"

#define ITERATIONS 100000

int main() {
#ifdef USE_ARENA
    arena_init(&arena, 64 * 1024 * 1024);  // 64MB arena
    printf("Using: ARENA allocator\n");
#else
    printf("Using: MALLOC allocator\n");
#endif

    Mat *a = mat_from(3, 3, (mat_elem_t[]){1,2,3, 4,5,6, 7,8,9});
    Mat *b = mat_from(3, 3, (mat_elem_t[]){9,8,7, 6,5,4, 3,2,1});

    clock_t start = clock();

    for (int i = 0; i < ITERATIONS; i++) {
        Mat *result = mat_rmul(a, b);
        mat_free_mat(result);  // no-op with arena, actual free with malloc
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Iterations: %d\n", ITERATIONS);
    printf("Time: %.3f seconds\n", elapsed);

#ifdef USE_ARENA
    printf("Arena used: %zu bytes\n", arena.offset);
    arena_free(&arena);
#endif

    mat_free_mat(a);
    mat_free_mat(b);
    return 0;
}
