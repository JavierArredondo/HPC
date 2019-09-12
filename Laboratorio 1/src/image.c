# include "../include/image.h"

/*TODO
- validation when memory can't be allocated
*/

image* readImage(FILE* file, int size) {
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (float**)malloc(sizeof(float*) * size);
	int i, j;
	for (i = 0; i < img->size; ++i)
	{
		img->content[i] = (float*)malloc(sizeof(float) * size);
		for (j = 0; j < img->size; ++j){
			int aux;
			fread(&aux, sizeof(float), 1, file);
			img->content[i][j] = aux ? 1.0 : 0.0;
		}
	}
	fclose(file);
	return img;
}

image* blankImage(int size, image* original){
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (float**)calloc(size, sizeof(float*));
	int i;
	for (i = 0; i < img->size; ++i){
		if(i == 0 || i == img->size-1)
			img->content[i] = (float*)realloc(original->content[i], sizeof(float) * img->size);
		else{
			img->content[i] = (float*)calloc(size, sizeof(float));
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
			printf("%f\n", img->content[i][j]);
			fwrite(&img->content[i][j], sizeof(int), 1, file);
		}
		/*for(j = 0; j < img->size; j++){
			if(j == img->size-1)
				fprintf(file, "%i", img->content[i][j]);
			else
				fprintf(file, "%i ", img->content[i][j]);
		}
		if(i != img->size-1)
			fprintf(file, "\n");*/
	}
}

int getUp(int x, int y, image* img){
	return x-1 >= 0 && img->content[x-1][y] == 1.0 ? 1 : 0;
}

int getRight(int x, int y, image* img){
	return y+1 < img->size && img->content[x-1][y] == 1.0 ? 1 : 0;
}

int getDown(int x, int y, image* img){
	return x+1 < img->size && img->content[x-1][y] == 1.0 ? 1 : 0;
}

int getLeft(int x, int y, image* img){
	return y-1 >= 0 && img->content[x-1][y] == 1.0 ? 1 : 0;
}

int getCenter(int x, int y, image* img){
	return img->content[x][y] == 1.0 ? 1 : 0;
}

void dilation_seq(image* input, image* result){
	int i, j, center;
	for(i = 1; i < input->size - 1; i++)
		for(j = 1; j < input->size - 1; j++)
			result->content[i][j] = getUp(i, j, input) || getRight(i, j, input) || getDown(i, j, input) || getLeft(i, j, input) || getCenter(i, j, input);
}

float* getUpSIMD(int x, int y, image* img){
	float* res __attribute__((aligned(16))) = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 0; i < 4; i++)
		res[i] = (x-1 >= 0 && y+i < img->size) ? img->content[x-1][y+i] : -1.0;
	return res;
}

float* getDownSIMD(int x, int y, image* img){
	float* res __attribute__((aligned(16))) = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 0; i < 4; i++)
		res[i] = (x+1 < img->size && y+i < img->size) ? img->content[x+1][y+i] : -1.0;
	return res;
}

float* getLeftSIMD(int x, int y, image* img){
	float* res __attribute__((aligned(16))) = (float*)calloc(4, sizeof(float));
	int i;
	for(i = -1; i < 3; i++)
		res[i] = (y+i >= 0 && y+i < img->size) ? img->content[x][y+i] : -1.0;
	return res;
}

float* getRightSIMD(int x, int y, image* img){
	float* res __attribute__((aligned(16))) = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 1; i < 5; i++)
		res[i] = (y+i >= 0 && y+i < img->size) ? img->content[x][y+i] : -1.0;
	return res;
}

float* getCenterSIMD(int x, int y, image* img){
	float* res __attribute__((aligned(16))) = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 0; i < 4; i++)
		res[i] = (y+i >= 0 && y+i < img->size) ? img->content[x][y+i] : -1.0;
	return res;
}

void dilation_simd(image* input, image* result){
	//__m128 up, right, down, left, center;
	__m128 R1, R2, center;
	int i, j;
	for(i = 0; i < input->size; i++){
		for(j = 0; j < input->size; j = j+4){
			R1 =  _mm_max_ps(_mm_load_ps(getDownSIMD(i, j, input)), _mm_load_ps(getUpSIMD(i, j, input)));
			R2 =  _mm_max_ps(_mm_load_ps(getRightSIMD(i, j, input)), _mm_load_ps(getLeftSIMD(i, j, input)));
			center =  _mm_max_ps(_mm_load_ps(getCenterSIMD(i, j, input)), _mm_max_ps(R1, R2));
			_mm_store_ps(&result->content[i][j], center);
		}
	}
}