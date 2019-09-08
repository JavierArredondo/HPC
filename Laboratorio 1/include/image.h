# ifndef IMAGE_H
# define IMAGE_H

typedef struct image {
	int size;
	char* filename;
	int** content;
} image;

image* readimage(FILE* file, int size);
void printImage(image* img);
void freeImage(image* img);


# endif