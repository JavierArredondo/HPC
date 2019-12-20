#include <stdio.h>


int** createMatrix(int n) {
	int** matrix = (int**)malloc(sizeof(int*) * n);
	for (int i = 0; i < n; ++i)
	{
		matrix[i] = (int*)malloc(sizeof(int) * n);
		for (int j = 0; j < n; ++j)
		{
			matrix[i][j] = rand() % 5;
		}
	}
	return matrix;
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

void mostrarFlat(int* a, int n){
	for(int i = 0; i < n*n; ++i) {
		printf("%i ", a[i]);
	}
}

int* toFlat(int** A, int n) {
	int* B = (int*)malloc(sizeof(int) * n * n);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < n; ++j) {
			B[i*n+j] = A[i][j];
		}
	}
	return B;
}

__global__ void vecAdd(int* a, int* b, int*c, int n) {
	// Obtención del Id global
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.x + threadIdx.x;
    printf("I'm in %ix%i (%i*%i + %i)\n", row, col, blockIdx.x, blockDim.x, threadIdx.x);
     // Nos aseguramos de no salir de los bordes
    if (row < n && col < n)
    {
    	int sum = 0;
    	for(int i = 0; i < n; ++i) {
    		
    		sum += a[row*n + i] * b[i * n + col];
    	}
    	c[row*n + col] = sum;	
    }
}

__host__ int main(void)
{
	/* PARTE DEL HOST: CREACIÓN DE MATRICES Y TO FLAT*/
	int n = 4;
	int ** A = createMatrix(n);
	int ** B = createMatrix(n);
	mostrar(A, n);
	printf("\n");
	mostrar(B, n);
	int* a = toFlat(A, n);
	int* b = toFlat(B, n);
	int* c = (int*)malloc(n*n*sizeof(int));
	mostrarFlat(a, n);
	printf("\n");
	mostrarFlat(b, n);
	printf("\n");
	/* CASTEAR PARA LLEVAR A DEVICE*/
	int *d_a;
    int *d_b;
    // Vector de salida del device
    int *d_c;
    // Se PIDE MEMORIA PARA VECTORES DE DEVICE A LO TAROT
    cudaMalloc(&d_a, n*n*sizeof(int));
    cudaMalloc(&d_b, n*n*sizeof(int));
    cudaMalloc(&d_c, n*n*sizeof(int));
    // Se copia el vector del host al vector del device
    cudaMemcpy( d_a, a, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, b, n*n*sizeof(int), cudaMemcpyHostToDevice);
    // CON ESTO "ESTA" EN DEVICE LOS VECTORSITOS: AHORA HAY QUE AJUSTAR LA GRILLA Y BLOQUES
    // Número de threads en cada bloque
    dim3 dimBlock(1,1);
    // Número de bloques en la grilla
   	dim3 dimGrid(n,n);//(int)ceil((float)n*n/blockSize);
    //printf("BS: %i \nGS: %i\n", blockSize, gridSize);


    vecAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n*n*sizeof(int), cudaMemcpyDeviceToHost );
    mostrarFlat(c, n);
	return 0;
}