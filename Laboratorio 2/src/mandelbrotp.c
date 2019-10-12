# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>
# ifdef _OPENMP
# include <omp.h>
# endif


/*Functions*/
double square(double number);
double distance(double x, double y);
int pixelsAxes(double upperLim, double lowerLim, double samples);
double** createMatrix(int rPix, int iPix);
void freeMatrix(double **matrix, int rows);
void writeImage(double** matrix, char* output, int rPixels, int iPixels);
long getMicrotime();
void mandelbrot(double** matrix, int rPixels, int iPixels, double rLowerLim, double rUpperLim, double iLowerLim, double iUpperLim, double samples, int depth, int threads);

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
	if (argc < 17){
		printf("The number of parameters less than requested\n");
		return 1;
	}
	else if(argc > 17){
		printf("The number of parameters greater than the requested\n");
		return 1;
	}
	/*Get values with getopt*/
	int i, t, cc;
	double a, b, c, d, s;
	char* f = (char*) malloc(sizeof(char) * 50);
	while ((cc = getopt (argc, argv, "i:a:b:c:d:s:f:t:")) != -1)
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
			case 't':
				sscanf(optarg, "%i", &t);
				break;
			case '?':
				if (optopt == 'i' || optopt == 'a' || optopt == 'b' || optopt == 'c' || optopt == 'd' || optopt == 's' || optopt == 'f' || optopt == 't' )
						printf("Option -%c required.\n", optopt);
				else if (isprint (optopt))
						printf("Option unknown `-%c'.\n", optopt);
				else
						printf("Unknown`\\x%x'.\n",optopt);
				return 1;
			default:
				abort();
			}
	int rPix = pixelsAxes(c, a, s);
	int iPix = pixelsAxes(d, b, s);
	double** matrix = createMatrix(rPix, iPix);
	// Print micro time at start
	printf("%ld\n", getMicrotime());
	mandelbrot(matrix, rPix, iPix, a, c, b, d, s, i, t);
	// Print micro time at end
	printf("%ld\n", getMicrotime());
	writeImage(matrix, f, rPix, iPix);
	freeMatrix(matrix, rPix);
	return 0;
}

void mandelbrot(double** matrix, int rPixels, int iPixels, double rLowerLim, double rUpperLim, double iLowerLim, double iUpperLim, double samples, int depth, int threads) {
	double Cx, Cy, Zn_r, Zn_i, Zn_aux;
	int n, i, r;
	# pragma omp parallel shared(matrix, rPixels, iPixels, rLowerLim, rUpperLim, samples, depth) private(Cx, Cy, Zn_r, Zn_i, Zn_aux, n, i, r) num_threads(threads)
	{
		# pragma omp for
		for(i = 0; i < iPixels; i++) {
			Cy = iLowerLim + samples * i;
			for (r = 0; r < rPixels; r++)
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
				//printf("%f with %i iterations and %.10f of distance M(%i,%i)->C(%.10f, %.10fi)\n", matrix[r][i], n, distance(Zn_r, Zn_i), r, i, Cx, Cy);
			}
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
