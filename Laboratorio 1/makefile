INC_DIRS = -Iinclude

CFLAGS = -c

main: obj/main.o
	@ gcc obj/*.o -o bin/main.out
	@ echo "Compilation."

obj/main.o: src/main.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/main.c -o obj/main.o
obj/image.o: src/image.c
	@ gcc  $(CFLAGS) $(INC_DIRS) src/image.c -o obj/image.o

clean:
	@ echo "Clean."
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "Running..."
	@ ./bin/main.out -i ./input/lena256x256.raw -s output/lena_seq.raw -p output/lena_simd.raw -N 256 -D 0

start: clean main run