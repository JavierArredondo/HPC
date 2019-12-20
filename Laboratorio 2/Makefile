INC_DIRS = -Iinclude

CFLAGS = -c

compilation:
	@ gcc src/mandelbrot.c -o bin/mandelbrot -lm
	@ gcc src/mandelbrotp.c -o bin/mandelbrotp -fopenmp -lm
	@ echo "Compilation."

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
	@ ./bin/mandelbrotp -i 500 -a -2 -b -1 -c 2 -d 1 -s 0.01 -f output/salida.raw -t 12

runp2:
	@ echo "Running...\n"
	@ ./bin/mandelbrotp -i 500 -a -0.748766713922161 -b 0.123640844894862 -c -0.748766707771757 -d 0.123640851045266 -s 1e-11 -f output/salida.raw -t 12

start: clean compilation run

startp: clean compilation runp