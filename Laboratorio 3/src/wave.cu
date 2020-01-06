# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <ctype.h>
# include <unistd.h>
# include <sys/time.h>
# define cc 1.0
# define dt 0.1
# define dd 2.0

float C1 = pow(cc, 2) * pow(dt/dd, 2) * 0.5;
float C2 = pow(cc, 2) * pow(dt/dd, 2);

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
			if(image[i*N + j] != 0) {
				printf("%f \n", image[i*N + j]);
			}
		}
	}
}


void write_image(int N, float* grid, char* filename) {

	FILE* file = fopen(filename, "wb");
	if(!file)
		return;
	fwrite(grid, N*N, sizeof(float), file);
	fclose(file);
}

__global__ void schroedinger(float* data, float* data1, float* data2, float* response, int N, int T, int t, float C1, float C2) {
	__shared__ float* H[3];
	int i, j;
 	/*First step: Sort precedence of Hx in function of t (iteration)*/
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		int shift = t%3;
		if(shift == 0) {
			H[0] = data;
			H[1] = data1;
			H[2] = data2;
		}
		else if(shift == 1) {
			H[0] = data2;
			H[1] = data;
			H[2] = data1;
		}
		else {
			H[0] = data1;
			H[1] = data2;
			H[2] = data;
		}
	}
	//printf("(%i, %i)  => [%i - %i]\n", threadIdx.x, threadIdx.y, threadIdx.x * blockDim.x, threadIdx.y * blockDim.y);
	__syncthreads();

	/*Second step: Compute Ht value*/
	for(i = threadIdx.x*blockDim.x ; i < threadIdx.x*blockDim.x + blockDim.x; ++i) {
		for(j = threadIdx.y*blockDim.y ; j < threadIdx.y*blockDim.y + blockDim.y; ++j) {
			if(i > 0 && i < N-1 && j > 0 && j < N-1) {
				float aux = H[1][(i+1)*N + j] + H[1][(i-1)*N + j] + H[1][i*N + (j+1)] + H[1][i*N + (j-1)] - 4*H[1][i*N + j];
				if(t == 1)
					H[0][i*N + j] = H[1][i*N + j] + C1 * aux;
				else
					H[0][i*N + j] = 2*H[1][i*N + j] - H[2][i*N + j] + C2 * aux;
				response[i*N + j] = H[0][i*N + j];
			}
		}
	}
	__syncthreads();
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
	printf("- C1: %f\n- C2: %f\n", C1, C2);


	int size = N*N;
	/* Vectors in host  */
	float* H_h;

	/* Vectors in device */
	float* H_0d;
	float* H_1d;
	float* H_2d;
	float* H_rd;

	/* Allocate memory for each vector in host */
	H_h = initial_condition(N);
	/* Allocate memory for each vector in device */
	cudaMalloc(&H_0d, size * sizeof(float));
	cudaMalloc(&H_1d, size * sizeof(float));
	cudaMalloc(&H_2d, size * sizeof(float));
	cudaMalloc(&H_rd, size * sizeof(float));

	/* Transfer values from host to device */
    cudaMemcpy(H_0d, H_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(H_1d, H_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(H_2d, H_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(H_rd, H_h, size * sizeof(float), cudaMemcpyHostToDevice);


    /* Define grid size and block size for device */
	dim3 dimBlock(x, y);
	dim3 dimGrid(1, 1);

	for(int iteration = 1; iteration < t; ++iteration) {
		schroedinger<<<dimGrid, dimBlock>>>(H_0d, H_1d, H_2d, H_rd, N, T, iteration, C1, C2);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(H_h, H_rd, size * sizeof(float), cudaMemcpyDeviceToHost);

	write_image(N, H_h, f);
	return 0;
}