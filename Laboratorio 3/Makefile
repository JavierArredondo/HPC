INC_DIRS = -Iinclude

CFLAGS = -c
NVCC = nvcc

compilation:
	@ echo "Compilation."
	@ nvcc src/wave.cu -o bin/wave

clean:
	@ echo "Clean."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "Running...\n"
	@ ./bin/wave -N 256 -x 16 -y 16 -T 10 -f output/salida.raw -t 300

start: clean compilation run