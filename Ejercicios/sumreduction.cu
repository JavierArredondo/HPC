# include <stdio.h>


int* createVector(int n){
	int* V = (int*)malloc(sizeof(int)*n);
	for(int i = 0; i < n; ++i) {
		V[i] = i;
		printf("%i ", V[i]);
	}
	printf("\n");
	return V;
}



__global__ void sumreduction(int *A, int N, int *sum ) {
	// Declare memoria compartida para el bloque
	printf("%i\n", threadIdx.x);
	int globaltid = blockIdx.x*blockDim.x + threadIdx.x; // ID global de la hebra
	int realN = N/2;
	
	while(realN != 0 && globaltid < realN) {
		printf("sum=%i\ntid=%i\n\n", A[globaltid]+A[globaltid+realN], globaltid);
		A[globaltid] = A[globaltid]+A[globaltid+realN];
		realN = realN / 2;
		printf("Ahora %i con %i\n", realN, A[globaltid]);
		__syncthreads();
	}
	__syncthreads();
	//sum = A;
	
	if (globaltid == 0) 
	{
		for(int i = 0; i < N; ++i) {
			printf("%i ", A[i]);
		}
		printf("\n");
		printf("%i\n", A[0]);
		atomicAdd(sum, A[0]);
	}
	// Cargar bloque de memoria compartida

	/*while(globaltid < realN && realN != 1)
	{
		temp[threadIdx.x] = A[globaltid]+A[globaltid+realN];
		realN = realN/2;
		printf("sum=%i\ntid=%i\n\n", A[globaltid]+A[globaltid+realN], globaltid);
		//__syncthreads();

	}*/



	
	//__syncthreads();

	/*for(int i = 0; i < realN; ++i) {
		
	}*/
	




	// Sincronizar a que todas hayan terminado
	// Reduccion iterativa dentro del bloque
	// Sincronizar a que todas haya terminado
	// Reduccion total a memoria global sum
	// printf("I'm in %i (%i*%i + %i)\n", globaltid, blockIdx.x, blockDim.x, threadIdx.x);
}






__host__ int main(void) {
	int N = 8;
	int* V = createVector(N);
	int B = 2; // blocksize
	int gridSize = (int)ceil((float)N/B);
	int* A;
	int sumita;
	int sum = 0;
	cudaMalloc(&A, sizeof(int) * N);
	cudaMalloc((void *) sum, sizeof(int));
	cudaMemcpy(A, V, sizeof(int)*N, cudaMemcpyHostToDevice);
	sumreduction<<<gridSize, B>>>(A, N, &sum);
	cudaDeviceSynchronize();
	cudaMemcpy(&sumita, sum, sizeof(int), cudaMemcpyDeviceToHost);
	printf("SUMITA %i\n", sumita);
	printf("SUMITA %i\n", sum);
}