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
	int** content;
} image;

image* readImage(FILE* file, int size);
image* blankImage(int size, image* original);
void printImage(image* img);
void fprintImage(image* img, FILE* file);
int getUp(int x, int y, image* img);
int getRight(int x, int y, image* img);
int getDown(int x, int y, image* img);
int getLeft(int x, int y, image* img);
int getCenter(int x, int y, image* img);
void dilation_seq(image* input, image* result);
void dilation_simd(image* input, image* result);
void freeImage(image* img);



# endif