INC_DIRS = -Iinclude

CFLAGS = -c

main: obj/main.o
	@ gcc obj/*.o -o bin/main.out
	@ echo "Compilation completed successfully"

obj/main.o: src/main.c
	@ echo "src/main.c"
	@ gcc  $(CFLAGS) $(INC_DIRS) src/main.c -o obj/main.o
obj/image.o: src/image.c
	@ echo "src/image.c"
	@ gcc  $(CFLAGS) $(INC_DIRS) src/image.c -o obj/image.o

clean:
	@ echo "Cleaning folders"
	@ rm -rf obj/*
	@ rm -rf bin/*
	@ rm -rf output/*

run:
	@ echo "ñkljh"