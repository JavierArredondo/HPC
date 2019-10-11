INC_DIRS = -Iinclude

CFLAGS = -c

compilation: obj/mandelbrot.o obj/mandelbrotp.o
	@ gcc obj/mandelbrot.o -o bin/mandelbrot -lm
	@ gcc obj/mandelbrotp.o -o bin/mandelbrotp -lm -fopenmp
	@ echo "Compilation."

obj/mandelbrot.o: src/mandelbrot.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/mandelbrot.c -o obj/mandelbrot.o

obj/mandelbrotp.o: src/mandelbrotp.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/mandelbrotp.c -o obj/mandelbrotp.o


clean:
	@ echo "Clean."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "Running...\n"
	@ ./bin/mandelbrot -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f output/salida.raw

runp:
	@ echo "Running...\n"
	@ ./bin/mandelbrotp -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f output/salida.raw -t 24

start: clean compilation run

startp: clean compilation runp