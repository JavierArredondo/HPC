# include "../include/image.h"

/*TODO
- validation when memory can't be allocated
*/

image* readImage(FILE* file, int size) {
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (int**)malloc(sizeof(int*) * size);
	int i, j;
	for (i = 0; i < img->size; ++i)
	{
		img->content[i] = (int*)malloc(sizeof(int) * size);
		for (j = 0; j < img->size; ++j){
			fscanf(file, "%d", &img->content[i][j]);
		}
	}
	fclose(file);
	return img;
}

image* blankImage(int size, image* original){
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (int**)calloc(size, sizeof(int*));
	int i;
	for (i = 0; i < img->size; ++i){
		if(i == 0 || i == img->size-1)
			img->content[i] = (int*)realloc(original->content[i], sizeof(int) * img->size);
		else{
			img->content[i] = (int*)calloc(size, sizeof(int));
			img->content[i][0] = original->content[i][0];
			img->content[i][img->size-1] = original->content[i][img->size-1];
		}
	}
	return img;
}

void printImage(image* img){
	int i,j;
	for(i = 0; i < img->size; i++){
		for(j = 0; j < img->size; j++){
			printf("%s", img->content[i][j]? "#" : ".");
			//printf("%i", img->content[i][j]);
		}
		printf("\n");
	}
}

void fprintImage(image* img, FILE* file){
	int i,j;
	for(i = 0; i < img->size; i++){
		for(j = 0; j < img->size; j++){
			if(j == img->size-1)
				fprintf(file, "%i", img->content[i][j]);
			else
				fprintf(file, "%i ", img->content[i][j]);
		}
		if(i != img->size-1)
			fprintf(file, "\n");
	}
}

int getUp(int x, int y, image* img){
	if(x-1 >= 0 && !img->content[x-1][y])
		return 1;
	return 0;
}

int getRight(int x, int y, image* img){
	if(y+1 < img->size && !img->content[x-1][y])
		return 1;
	return 0;
}

int getDown(int x, int y, image* img){
	if(x+1 < img->size && !img->content[x-1][y])
		return 1;
	return 0;
}

int getLeft(int x, int y, image* img){
	if(y-1 >= 0 && !img->content[x-1][y])
		return 1;
	return 0;
}
void dilation(image* input, image* result){
	int i, j, center;
	for(i = 1; i < input->size - 1; i++){
		for(j = 1; j < input->size - 1; j++){
			//result->content[i][j] = getUp(i, j, input) + getRight(i, j, input) + getDown(i, j, input) + getLeft(i, j, input);
			result->content[i][j] = !(getUp(i, j, input) || getRight(i, j, input) || getDown(i, j, input) || getLeft(i, j, input) || !input->content[i][j]);
		}
	}
}