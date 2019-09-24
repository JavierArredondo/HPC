INC_DIRS = -Iinclude

CFLAGS = -c

compilation: obj/dilation.o
	@ gcc obj/*.o -o bin/dilation
	@ echo "Compilation."

obj/dilation.o: src/dilation.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/dilation.c -o obj/dilation.o
obj/image.o: src/image.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/image.c -o obj/image.o

clean:
	@ echo "Clean."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "Running...\n"
	@ ./bin/dilation -i ./input/pikachu256x256.raw -s output/pikachu_seq.raw -p output/pikachu_simd.raw -N 256 -D 1

start: clean compilation run