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

clean:
	rm -f add_mat sub_mat transpose_mat reshape_mat mat_log
