# include "../include/image.h"

/*
Params: - Pointer to file: File to read content. Must be a raw image.
		- Integer size: Size of side of image. NxN

Return: - Pointer to structure of image.
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
			int aux;
			fread(&aux, sizeof(int), 1, file);
			img->content[i][j] = aux? 1 : 0;
		}
	}
	fclose(file);
	return img;
}

/*
Params: - Integer size: Size of side of image. NxN
		- Pointer to structure of image.

Return: - Pointer to structure of blank image.
*/
image* blankImage(int size, image* original){
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (int**)calloc(size, sizeof(int*));
	int i;
	for (i = 0; i < img->size; i++)
			img->content[i] = (int*)calloc(size, sizeof(int));
	return img;
}

/*
Print image for stdout
Params: - Pointer to structure of image.
*/
void printImage(image* img){
	int i,j;
	for(i = 0; i < img->size; i++){
		for(j = 0; j < img->size; j++){
			printf("%s", img->content[i][j]? "#" : ".");
			//printf("%i ", img->content[i][j]);
		}
		printf("\n");
	}
}

/*
Write image in a file
Params: - Pointer to structure of image.
*/
void fprintImage(image* img, FILE* file){
	int i,j;
	for(i = 0; i < img->size; i++){
		for(j = 0; j < img->size; j++)
		{
			int aux = img->content[i][j] ? 255 : 0;
			fwrite(&aux, sizeof(int), 1, file);
		}
	}
	fclose(file);
}

/*
Params: - Position of pixel in x axis
		- Position of pixel in y axis
		- Pointer to structure of image.

Return: If position up, down, left, right or center is available to do dilation.
*/
int getUp(int x, int y, image* img){
	return img->content[x-1][y];
}

int getRight(int x, int y, image* img){
	return img->content[x][y+1];
}

int getDown(int x, int y, image* img){
	return img->content[x+1][y];
}

int getLeft(int x, int y, image* img){
	return img->content[x][y-1];
}

int getCenter(int x, int y, image* img){
	return img->content[x][y];
}


/*
Do sequential dilation
Params: - Pointer to structure of input image.
		- Pointer to structure of output image.
*/
void dilation_seq(image* input, image* result){
	if(!input && !result)
		perror("Input image or copy is wrong");
	int i, j;
	for(i = 1; i < input->size - 1; i++)
		for(j = 1; j < input->size - 1; j++)
			result->content[i][j] = getUp(i, j, input) || getRight(i, j, input) || getDown(i, j, input) || getLeft(i, j, input) || getCenter(i, j, input);
}

/*
Do SIMD dilation
Params: - Pointer to structure of input image.
		- Pointer to structure of output image.
*/
void dilation_simd(image* input, image* result){
	if(!input && !result)
		perror("Input image or copy is wrong");
	__m128i R1, R2, center;
	int i, j;
	int residue = input->size % 4;
	for(i = 1; i < input->size-1; i++){
		for(j = 1; j < input->size-4; j = j+4){
			R1 = _mm_max_epi16(_mm_loadu_si128((__m128i*)&input->content[i-1][j]), _mm_loadu_si128((__m128i*)&input->content[i+1][j]));
			R2 = _mm_max_epi16(_mm_loadu_si128((__m128i*)&input->content[i][j-1]), _mm_loadu_si128((__m128i*)&input->content[i][j+1]));
			center = _mm_max_epi16(_mm_loadu_si128((__m128i*)&input->content[i][j]), _mm_max_epi16(R1, R2));
			_mm_storeu_si128((__m128i*)&result->content[i][j], center);
		}
	}
	for(i = 1; i < input->size-1; i++)
		for(j = input->size-residue-4; j < input->size-1; j++)
			result->content[i][j] = getUp(i, j, input) || getRight(i, j, input) || getDown(i, j, input) || getLeft(i, j, input) || getCenter(i, j, input);
}

void freeImage(image* img){
	int i;
	for (i = 0; i < img->size; ++i)
		free(img->content[i]);
	free(img->content);
}