CC=g++
CFLAGS=-std=c++11

all: serial

serial: serial.cpp
	$(CC) $(CFLAGS) -o serial serial.cpp
	./serial $(n) $(t)

openmp: openmp.cpp
	$(CC) $(CFLAGS) -o openmp openmp.cpp -fopenmp
	./openmp $(n) $(t)

pthread: pthreads.cpp
	$(CC) $(CFLAGS) -o pthread pthreadnew.cpp -lpthread
	./pthread $(n) $(t)

.PHONY: clean
clean:
	rm -f serial openmp pthread