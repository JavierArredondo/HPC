INC_DIRS = -Iinclude

CFLAGS = -c
NVCC = nvcc

compilation:
	@ echo "Compilation."
	@ nvcc src/wave.cu -o bin/wave

info:
	@ nvcc -arch=sm_50 -Xptxas="-v" src/wave.cu -o bin/wave

clean:
	@ echo "Clean."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "Running...\n"
	@ ./bin/wave -N 256 -x 32 -y 16 -T 300 -f output/salida -t 10

start: clean compilation run