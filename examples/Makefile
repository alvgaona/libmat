CC = gcc
CFLAGS = -I..

add_mat: add_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

sub_mat: sub_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

transpose_mat: transpose_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

reshape_mat: reshape_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

mat_log: mat_log.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

eye_mat: eye_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

ones_mat: ones_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

zeros_mat: zeros_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

diag_mat: diag_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

vec_mat: vec_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

from_mat: from_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

at_mat: at_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

size_mat: size_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

get_diag_mat: get_diag_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

t_mat: t_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

add_inplace_mat: add_inplace_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

sub_inplace_mat: sub_inplace_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

rreshape_mat: rreshape_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

add_diag_mat: add_diag_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

add_vec_mat: add_vec_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

equals_mat: equals_mat.c ../mat.c ../mat_util.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f add_mat sub_mat transpose_mat reshape_mat mat_log eye_mat ones_mat zeros_mat diag_mat vec_mat from_mat at_mat size_mat get_diag_mat t_mat add_inplace_mat sub_inplace_mat rreshape_mat add_diag_mat add_vec_mat equals_mat
