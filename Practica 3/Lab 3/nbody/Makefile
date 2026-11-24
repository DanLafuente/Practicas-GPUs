


CC=gcc

CFLAGS=-O3 -lm -lrt

PGCC=pgcc -Minfo -fast -acc -ta=nvidia

NAME=nbody

default: host gpu

host:
	$(CC) $(NAME).c -o $(NAME).host.exe $(CFLAGS)

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe 

clean: 
	rm -f _ *.exe
