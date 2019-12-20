# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>
# ifdef _OPENMP
# include <omp.h>
# endif
# define upper 100
# define t 10

int** createMatrix(int n) {
	int** matrix = (int**)malloc(sizeof(int*) * n);
	for (int i = 0; i < n; ++i)
	{
		matrix[i] = (int*)malloc(sizeof(int) * n);
		for (int j = 0; j < n; ++j)
		{
			matrix[i][j] = rand() % 100;
		}
	}
	return matrix;
}

int** suma(int** A, int** B, int n) {
	int i, j;
	int** C = createMatrix(n);
	# pragma omp parallel shared(n, C) private(i, j) num_threads(t) 
	{
		# pragma omp for
		for (i = 0; i < n; i++)
		{

			for (j = 0; j < n; j++)
			{
				printf("Hello World!, numero de hebra = %d en %i, %i\n", omp_get_thread_num(), i, j);
				C[i][j] = A[i][j] + B[i][j];
			}
		}
	}
	return C;
}

int sumaTodo(int** A, int n) {
	int i, j, r = 0;
	# pragma omp parallel shared(n, A, r) private(i, j) num_threads(t) 
	{
		# pragma omp for reduction (+: r)
		for (i = 0; i < n; i++)
		{
			int r2 = 0;
			# pragma omp parallel shared(r2) num_threads(2)
			{
				# pragma omp for reduction(+: r2)
				for (j = 0; j < n; ++j)
				{
					r2 +=  A[i][j];
				}
				printf("R2 = %i thread %i\n", r2, omp_get_thread_num());
			}
			r+=r2;
		}
	}
	return r;
}

void mostrar(int** A, int n) {
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%3i ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char const *argv[])
{
	omp_set_nested(1);
	int n = 10;
	int** A = createMatrix(n);
	int** B = createMatrix(n);
	mostrar(A, n);
	mostrar(B, n);
	int** C = suma(A, B, n);
	mostrar(C, n);
	printf("Suma todo de A: %i\n", sumaTodo(A, n));
	return 0;
}