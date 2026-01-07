CC = gcc
CFLAGS = -I..

add_mat: add_mat.c
	$(CC) $(CFLAGS) -o $@ $^

sub_mat: sub_mat.c
	$(CC) $(CFLAGS) -o $@ $^

transpose_mat: transpose_mat.c
	$(CC) $(CFLAGS) -o $@ $^

reshape_mat: reshape_mat.c
	$(CC) $(CFLAGS) -o $@ $^

mat_log: mat_log.c
	$(CC) $(CFLAGS) -o $@ $^

eye_mat: eye_mat.c
	$(CC) $(CFLAGS) -o $@ $^

ones_mat: ones_mat.c
	$(CC) $(CFLAGS) -o $@ $^

zeros_mat: zeros_mat.c
	$(CC) $(CFLAGS) -o $@ $^

diag_mat: diag_mat.c
	$(CC) $(CFLAGS) -o $@ $^

vec_mat: vec_mat.c
	$(CC) $(CFLAGS) -o $@ $^

from_mat: from_mat.c
	$(CC) $(CFLAGS) -o $@ $^

at_mat: at_mat.c
	$(CC) $(CFLAGS) -o $@ $^

size_mat: size_mat.c
	$(CC) $(CFLAGS) -o $@ $^

get_diag_mat: get_diag_mat.c
	$(CC) $(CFLAGS) -o $@ $^

t_mat: t_mat.c
	$(CC) $(CFLAGS) -o $@ $^

add_inplace_mat: add_inplace_mat.c
	$(CC) $(CFLAGS) -o $@ $^

sub_inplace_mat: sub_inplace_mat.c
	$(CC) $(CFLAGS) -o $@ $^

rreshape_mat: rreshape_mat.c
	$(CC) $(CFLAGS) -o $@ $^

add_diag_mat: add_diag_mat.c
	$(CC) $(CFLAGS) -o $@ $^

add_vec_mat: add_vec_mat.c
	$(CC) $(CFLAGS) -o $@ $^

equals_mat: equals_mat.c
	$(CC) $(CFLAGS) -o $@ $^

copy_mat: copy_mat.c
	$(CC) $(CFLAGS) -o $@ $^

deep_copy_mat: deep_copy_mat.c
	$(CC) $(CFLAGS) -o $@ $^

scale_mat: scale_mat.c
	$(CC) $(CFLAGS) -o $@ $^

rscale_mat: rscale_mat.c
	$(CC) $(CFLAGS) -o $@ $^

add_scalar_mat: add_scalar_mat.c
	$(CC) $(CFLAGS) -o $@ $^

radd_scalar_mat: radd_scalar_mat.c
	$(CC) $(CFLAGS) -o $@ $^

rmul_mat: rmul_mat.c
	$(CC) $(CFLAGS) -o $@ $^

mul_mat: mul_mat.c
	$(CC) $(CFLAGS) -o $@ $^

hadamard_mat: hadamard_mat.c
	$(CC) $(CFLAGS) -o $@ $^

rhadamard_mat: rhadamard_mat.c
	$(CC) $(CFLAGS) -o $@ $^

add_many_mat: add_many_mat.c
	$(CC) $(CFLAGS) -o $@ $^

radd_many_mat: radd_many_mat.c
	$(CC) $(CFLAGS) -o $@ $^

dot_mat: dot_mat.c
	$(CC) $(CFLAGS) -o $@ $^

matdef_mat: matdef_mat.c
	$(CC) $(CFLAGS) -o $@ $^

strip_mat: strip_mat.c
	$(CC) $(CFLAGS) -o $@ $^

arena_mat: arena_mat.c
	$(CC) $(CFLAGS) -DUSE_ARENA -o $@ $^

arena_mat_malloc: arena_mat.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f add_mat sub_mat transpose_mat reshape_mat mat_log eye_mat ones_mat zeros_mat diag_mat vec_mat from_mat at_mat size_mat get_diag_mat t_mat add_inplace_mat sub_inplace_mat rreshape_mat add_diag_mat add_vec_mat equals_mat copy_mat deep_copy_mat scale_mat rscale_mat add_scalar_mat radd_scalar_mat rmul_mat mul_mat hadamard_mat rhadamard_mat add_many_mat radd_many_mat dot_mat matdef_mat strip_mat arena_mat arena_mat_malloc
