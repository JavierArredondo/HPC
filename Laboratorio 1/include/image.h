# ifndef IMAGE_H
# define IMAGE_H

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <ctype.h>
# include <unistd.h>
# include <pmmintrin.h>
# include <immintrin.h>

typedef struct image {
	int size;
	char* filename;
	float** content;
} image;

image* readImage(FILE* file, int size);
void printImage(image* img);
void freeImage(image* img);


# endif