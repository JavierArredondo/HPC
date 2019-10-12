# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>

double square(double number);
double distance(double x, double y);
int pixelsAxes(double upperLim, double lowerLim, double samples);
double** createMatrix(int rPix, int iPix);
void freeMatrix(double **matrix, int rows);
void writeImage(double** matrix, char* output, int rPixels, int iPixels);
long getMicrotime();
void mandelbrot(double** matrix, int rPixels, int iPixels, double rLowerLim, double rUpperLim, double iLowerLim, double iUpperLim, double samples, int depth);

int main(int argc, char *argv[])
{
	/*Parameters of the program: 
	-i     depth, number of iterations
	-a:    limite interior del componente real del plano complejo
	-b:    limite inferior componente iaginario
	-c:    limite superior componente real
	-d:    limite superior componetne imaginario
	-s:    sampleo
	-f:    archivo de salida
	*/
	opterr = 0;
	//printf("Parameters %i\n", argc);
	if (argc < 15){
		printf("The number of parameters less than requested\n");
		return 1;
	}
	else if(argc > 15){
		printf("The number of parameters greater than the requested\n");
		return 1;
	}
	/*Get values with getopt*/
	int i, cc;
	double a, b, c, d, s;
	char* f = (char*) malloc(sizeof(char) * 50);
	
	while ((cc = getopt (argc, argv, "i:a:b:c:d:s:f:")) != -1)
		switch (cc) {
			case 'i':
				sscanf(optarg, "%d", &i);
				break;
			case 'a':
				a = atof(optarg);
				break;
			case 'b':
				b = atof(optarg);
				break;
			case 'c':
				c = atof(optarg);
				break;
			case 'd':
				d = atof(optarg);
				break;
			case 's':
				s = atof(optarg);
				break;
			case 'f':
				sscanf(optarg, "%s", f);
				break;
			case '?':
				if (optopt == 'i' || optopt == 'a' || optopt == 'b' || optopt == 'c' || optopt == 'd' || optopt == 's' || optopt == 'f')
						printf("Option -%c required.\n", optopt);
				else if (isprint (optopt))
						printf("Option unknown `-%c'.\n", optopt);
				else
						printf("Unknown`\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
			}
	//printf("- i: %i\n- a: %f\n- b: %f\n- c: %f\n- d: %f\n- s: %f\n- f: %s\n", i, a, b, c, d, s, f);
	int rPix = pixelsAxes(c, a, s);
	int iPix = pixelsAxes(d, b, s);
	double** matrix = createMatrix(rPix, iPix);
	printf("%ld\n", getMicrotime());
	mandelbrot(matrix, rPix, iPix, a, c, b, d, s, i);
	printf("%ld\n", getMicrotime());
	writeImage(matrix, f, rPix, iPix);
	freeMatrix(matrix, rPix);
	return 0;
}


void mandelbrot(double** matrix, int rPixels, int iPixels, double rLowerLim, double rUpperLim, double iLowerLim, double iUpperLim, double samples, int depth) {
	double Cx, Cy, Zn_r, Zn_i, Zn_aux;
	int n;
	for(int i = 0; i < iPixels; i++) {
		Cy = iLowerLim + samples * i;
		for (int r = 0; r < rPixels; r++)
		{
			Cx = rLowerLim + samples * r;
			n = 1;
			Zn_r = Cx;
			Zn_i = Cy;
			while( distance(Zn_r, Zn_i) < 2 && n < depth) {
				
				Zn_aux = square(Zn_r) + Cx - square(Zn_i);
				Zn_i = 2*Zn_i*Zn_r + Cy;
				Zn_r = Zn_aux;		
				n++;
			}
			matrix[r][i] = log(n) + 1;
		}
	}
}

double square(double number){
	return number * number;
}

double distance(double x, double y) {
	return sqrt( square(x) + square(y) );
}

/* Calculate distance between 2 points (ru, 0) and (rl, 0) equals to abs(ru-rl)
   Cells between two points; distance/samples + 1, because we need to add a last point of the cell in the sample 
*/
int pixelsAxes(double upperLim, double lowerLim, double samples){
	return rint(fabs(upperLim - lowerLim)/samples) + 1;
}

double** createMatrix(int rPix, int iPix) {
	double** matrix = (double**)malloc(sizeof(double*) * rPix);
	for (int i = 0; i < rPix; ++i)
	{
		matrix[i] = (double*)malloc(sizeof(double) * iPix);
	}
	return matrix;
}

void writeImage(double** matrix, char* output, int rPixels, int iPixels) {
	FILE* file = fopen(output, "wb");
	if(!file)
		return;
	float *aux = (float*)malloc(sizeof(float) * rPixels);
	for (int i = 0; i < rPixels; ++i)
	{
		for(int j = 0; j < iPixels; j++)
			aux[j] = (float)matrix[i][j];
		fwrite(aux, sizeof(float), iPixels, file);
	}
	fclose(file);
}

void freeMatrix(double** mat, int rows){
	int i;
	for (i = 0; i < rows; ++i)
		free(mat[i]);
	free(mat);
}

/*Get from https://gist.github.com/sevko/d23646ba07c77c15fde9*/
long getMicrotime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}
