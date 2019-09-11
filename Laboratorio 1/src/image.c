# include "../include/image.h"

/*TODO
- validation when memory can't be allocated
*/

image* readImage(FILE* file, int size) {
	image *img = (image*)malloc(sizeof(image));
	img->size = size;
	img->content = (float**)malloc(sizeof(float*) * size);
	int i, j, c;
	for (i = 0; i < img->size; ++i)
	{
		img->content[i] = (float*)malloc(sizeof(float) * size);
		for (j = 0; j < img->size; ++j){
			//fscanf(file, "%d", &img->content[i][j]);
			float a;
			fread(&a, sizeof(float), 1, file);
			if(a < 0)
				img->content[i][j] = 0.0;
			else
				img->content[i][j] = 1.0;
			c++;
			//printf("%i ", img->content[i][j]);
		}
	}
	printf("Count %i\n", c);
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
		fwrite(img->content[i], sizeof(float), img->size, file);
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
	if(x-1 >= 0 && img->content[x-1][y] == 1.0)
		return 1;
	return 0;
}

int getRight(int x, int y, image* img){
	if(y+1 < img->size && img->content[x-1][y] == 1.0)
		return 1;
	return 0;
}

int getDown(int x, int y, image* img){
	if(x+1 < img->size && img->content[x-1][y] == 1.0)
		return 1;
	return 0;
}

int getLeft(int x, int y, image* img){
	if(y-1 >= 0 && img->content[x-1][y] == 1.0)
		return 1;
	return 0;
}

int getCenter(int x, int y, image* img){
	return img->content[x][y] == 1.0 ? 1 : 0;
}

void dilation_seq(image* input, image* result){
	int i, j, center;
	for(i = 1; i < input->size - 1; i++){
		for(j = 1; j < input->size - 1; j++){
			result->content[i][j] = getUp(i, j, input) || getRight(i, j, input) || getDown(i, j, input) || getLeft(i, j, input) || getCenter(i, j, input);
		}
	}
}

float* getUpSIMD(int x, int y, image* img){
	float* res = (float*)calloc(4, sizeof(float));
	for(i = 0; i < 4; i++)
		res[i] = (y-1 < 0 && x + i < img->size-1) ? 0.0 : 1.0;
	return res;
}

// Quizas no va el menos 1 en el size
float* getDownSIMD(int x, int y, image* img){
	float* res = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 0; i < 4; i++)
		res[i] = (y+1 < img->size-1 && x + i < img->size -1) ? 0.0 : 1.0;
	return res;
}

float* getDownSIMD(int x, int y, image* img){
	float* res = (float*)calloc(4, sizeof(float));
	int i;
	for(i = 0; i < 4; i++)
		res[i] = (y+1 < img->size-1 && x + i < img->size -1) ? 0.0 : 1.0;
	return res;
}

void dilation_simd(image* input, image* result){
	__m128 up, right, down, left, center;
	float buff[4] __attribute__((aligned(16)));
	int i, j;
	for(i = 0; i < input->size; i = i+4)
		for(j = 0; j < input->size; j = j+4){
			printf("(%i, %i) -> ", i, j);
			up = 
			center =  _mm_load_ps(&input->content[i][j]);
			float* wea = getUpSIMD(i, j, input);

			printf("CENTER points [%f %f %f %f]\n", center[0], center[1], center[2], center[3]);
			printf("UP points[%f %f %f %f]\n\n", wea[0], wea[1], wea[2], wea[3]);
			break;
		}

	//up = _mm_setr_si128(b[0], b[1], b[2], b[3]);
}