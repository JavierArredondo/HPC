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
	@ ./bin/main.out -i ./input/example14x14.raw -s output/example_seq.raw -p output/example_simd.raw -N 14 -D 1

start: clean main run