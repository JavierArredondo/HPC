# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>
# define cc 1.0
# define dt 0.1
# define dd 2.0

float* initial_condition(int N) {
	float* H = (float*)malloc(sizeof(float) * N * N);
	int i, j;
	for(i = 0; i < N; ++i) {
		for(j = 0; j < N; ++j) {
			if(0.4*N < i && i  < 0.6*N && 0.4*N < j && j < 0.6*N)
				H[i*N + j] = 20.0;
			else
				H[i*N + j] = 0.0;
		}
	}
	return H;
}

void show_image(int N, float* image) {
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < N; ++j) {
			printf("%0.1f ", image[i*N + j]);
		}
		printf("\n");
	}
}

__global__ void schroedinger(float* data, int N, int T, int t) {
	__shared__ float* H_0, H_1, H_2;
	int size = N*N;
	__syncthreads();
	int lower = (threadIdx.x * N + threadIdx.y * blockDim.y * N);
	int upper = (threadIdx.x * N + threadIdx.y * blockDim.y * N) + N;
	printf("(%i, %i) _%i %i _=> [%i - %i]\n", threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, lower, upper);


	/*int iteration, i, j;
	for(iteration = 0; iteration < t; ++iteration) {
		
		if(t == 0){
			for(i = N*threadIdx.x + threadIdx.y; i < N; ++i) {
				for(j = 0; j < N; ++j) {
					
				}
			}
		}
		else if(t == 1) {

		}
		else {

		}
	}*/





	//int id = blockIdx.x*blockDim.x + threadIdx.x;
	/*int i, j;
	for(i = 0; i < N-1; i=i+blockDim.x) {
		for(j = 0; j < N-1; j=j+blockDim.y) {
			/* code */
		//}
	//}

	__syncthreads();


	/*float aux;
	aux = pow(cc, 2) * pow(dt/dd, 2) * (H_1[(i+1)*N + j] + H_1[(i-1)*N + j] + H_1[i*N + (j+1)] + H_1[i*N + (j-1)] - 4*H_1[i*N + j]);
	if(t == 1)
		H[i*N + j] = H_1[i*N + j] + 0.5 * aux;
	else
		H[i*N + j] = 2 * H_1[i*N + j] - H_2[i*N + j] + aux;*/
}

__host__ int main(int argc, char *argv[]) {
	/*Parameters of the program: 
	-N    grid size
	-x    block size x
	-y    block size y
	-T    number of steps
	-f    output file
	-t    output iteration
	*/

	printf("Parameters %i\n", argc);
	char* f = (char*) malloc(sizeof(char) * 50);
	int N, x, y, T, t, c;

	opterr = 0;
	while ((c = getopt (argc, argv, "N:x:y:T:f:t:")) != -1)
		switch (c) {
			case 'N':
				sscanf(optarg, "%d", &N);
				break;
			case 'x':
				sscanf(optarg, "%d", &x);
				break;
			case 'y':
				sscanf(optarg, "%d", &y);
				break;
			case 'T':
				sscanf(optarg, "%d", &T);
				break;
			case 'f':
				sscanf(optarg, "%s", f);
				break;
			case 't':
				sscanf(optarg, "%d", &t);
				break;
			case '?':
				if (optopt == 'N' || optopt == 'x' || optopt == 'y' || optopt == 'T' || optopt == 'f' || optopt == 't')
						printf("Option -%c required.\n", optopt);
				else if (isprint (optopt))
						printf("Option unknown `-%c'.\n", optopt);
				else
						printf("Unknown`\\x%x'.\n", optopt);
				return 1;
			default:
				abort();
		}

	printf("- N: %i\n- x: %d\n- y: %d\n- T: %d\n- f: %s\n- t: %d\n", N, x, y, T, f, t);


	int size = N*N;
	/* Vectors in host  */
	float* H_0h;


	/* Vectors in device */
	float* H_0d;


	/* Allocate memory for each vector in host */
	H_0h = initial_condition(N);

	/* Allocate memory for each vector in device */
	cudaMalloc(&H_0d, size * sizeof(float));

	/* Transfer values from host to device */
    cudaMemcpy(H_0d, H_0h, size * sizeof(float), cudaMemcpyHostToDevice);

    /* Define grid size and block size for device */
	dim3 dimBlock(x, y);
	dim3 dimGrid(1, 1);

	schroedinger<<<dimGrid, dimBlock>>>(H_0d, N, T, t);
	cudaDeviceSynchronize();
	cudaMemcpy(H_0h, H_0d, size * sizeof(float), cudaMemcpyDeviceToHost);

	return 0;
}