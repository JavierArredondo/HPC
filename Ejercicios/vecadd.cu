#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>

// CUDA kernel. Cada thread ejecuta la operación sobrte un elemencto de c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Obtención del Id global
    int id = blockIdx.x*blockDim.x+threadIdx.x;
     // Nos aseguramos de no salir de los bordes
    if (id < n)
        c[id] = a[id] + b[id];
    printf("I am thread (%d, %d) in block (%d, %d). Global thread ID = %d\n",threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x);
    
}
 
__host__ int main( int argc, char* argv[] )
{
     
    // Tamaño de los vectores
    int n = 100000;
 
    // Vectores de entrada al host (CPU)
    double *h_a;
    double *h_b;
    // Vector de salida del host
    double *h_c;
 
    // Vector de entrada del device (GPU)
    double *d_a;
    double *d_b;
    // Vector de salida del device
    double *d_c;
 
    // Size, in bytes, of each vector
    //size_t bytes = n*sizeof(double);
 
    // Se asigna memoria para cada vector del host
    h_a = (double*)malloc(n*sizeof(double));
    h_b = (double*)malloc(n*sizeof(double));
    h_c = (double*)malloc(n*sizeof(double));
 
    // Se asigna memoria para cada vector del device
    cudaMalloc(&d_a, n*sizeof(double));
    cudaMalloc(&d_b, n*sizeof(double));
    cudaMalloc(&d_c, n*sizeof(double));
 
    int i;
    // Se inicializa los vectores del host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
 
    // Se copia el vector del host al vector del device
    cudaMemcpy( d_a, h_a, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Número de threads en cada bloque
    blockSize = 1024;
 
    // Número de bloques en la grilla
    gridSize = (int)ceil((float)n/blockSize);
    printf("%i\n", gridSize);
 
    // Se ejecuta el kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Se copia el vector resultante del device al host
    cudaMemcpy( h_c, d_c, n*sizeof(double), cudaMemcpyDeviceToHost );

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final : %f\n", sum/n);
 
    // Se libera la memoria del device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Se libera la memoria del host
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}