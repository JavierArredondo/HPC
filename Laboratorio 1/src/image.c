# include "../include/image.h"

image* readimage(FILE* file, int size) {
	image img;
	img.size = size;
	img.content = (int**)malloc(sizeof(int*) * size);
	for (int i = 0; i < img.size; ++i)
	{
		img.content[i] = (int*)malloc(sizeof(int) * size);
		for (int j = 0; j < img.size; ++j)
			fscanf(file, "%d", img.content[i][j])
	}
	return &img;
}