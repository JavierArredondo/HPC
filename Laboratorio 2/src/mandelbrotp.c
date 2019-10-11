# include <time.h>
# include "../include/mandelbrotp.h"

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


float** createMatrix(int rPix, int iPix) {
	float ** matrix = (float**)malloc(sizeof(float*) * iPix);
	for (int i = 0; i < iPix; ++i)
	{
		// TODO: Verify null pointer
		matrix[i] = (float*)malloc(sizeof(float) * rPix);
	}
	return matrix;
}

void mandelbrot(float** matrix, int rPixels, int iPixels, int rLowerLim, int rUpperLim, int iLowerLim, int iUpperLim, double samples, int depth, int threads) {
	omp_set_num_threads(threads);
	double Cx, Cy, Zn_r, Zn_i, Zn_aux;
	int n, i, r;
	# pragma omp parallel shared(matrix, rPixels, iPixels, rLowerLim, rLowerLim, samples, depth) private(Cx, Cy, Zn_r, Zn_i, Zn_aux, n, i, r)
	{
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
			}
		}
	}
}

void writeImage(float** matrix, char* output, int rPixels, int iPixels) {
	FILE* file = fopen(output, "wb");
	if(!file)
		return;
	for (int i = 0; i < iPixels; ++i)
	{
		fwrite(matrix[i], sizeof(float), rPixels, file);
	}
	fclose(file);
}

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
	printf("Parameters %i\n", argc);
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
	float a, b, c, d, s;
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
	printf("- i: %i\n- a: %f\n- b: %f\n- c: %f\n- d: %f\n- s: %f\n- f: %s\n- t: %i\n", i, a, b, c, d, s, f, t);

	int rPix = pixelsAxes(c, a, s);
	int iPix = pixelsAxes(d, b, s);
	printf("r %i i %i\n", rPix, iPix);
	float** matrix = createMatrix(rPix, iPix);
	mandelbrot(matrix, rPix, iPix, a, c, b, d, s, i, t);
	writeImage(matrix, f, rPix, iPix);
	return 0;
}