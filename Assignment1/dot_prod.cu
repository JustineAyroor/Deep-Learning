// NAME : JUSTINE J AYROOR
// UCID : ja573
// Assignment 1

// import libraries
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
// Kernel Function
__global__ void dotPro(float *a, float *b, float *c, int rows, int cols){

        int sum = 0;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(int i = 0; i < cols; i++){
		sum += a[i*rows+tid]*b[i]; 
	}	
	c[tid] = sum;
        return;
}

// Main Function
int  main(int argc, char* argv[]){
	//intialize Variables
	int rows,cols,CUDA_DEVICE = 0;
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        CUDA_DEVICE = atoi(argv[3]);
	int THREADS = atoi(argv[6]);
	int BLOCKS;
	int jobs;
        printf("Rows=%d Cols=%d  CUDA_DEVICE=%d\n",rows,cols,CUDA_DEVICE);
	
	// Set CUDA Device
        cudaError err = cudaSetDevice(CUDA_DEVICE);
        if(err != cudaSuccess) { 
		printf("Error setting CUDA DEVICE\n"); 
		exit(EXIT_FAILURE);
	 }
    	// Read Data File & Store in Array
	FILE *myFile;
    	myFile = fopen(argv[4], "r");
	float *a = (float *)malloc(cols*rows*sizeof(float));
	float *b = (float *)malloc(cols*sizeof(float));
	float c[rows];
	
    	int i,j;
	if((myFile !=(FILE*)NULL)){
		for(int i = 0;i<rows*cols;i++)
			fscanf(myFile,"%f", &a[(i%cols)*rows+(int)(i/cols)]);		
	}else{
		printf("Could not open data file");
	}
	fclose(myFile);
	
	// Read Vector File & Store in Array 
    	FILE *mywFile;
	mywFile = fopen(argv[5], "r");
    	//int b[cols];
    	for(i = 0; i < cols; i++){
		fscanf(mywFile, "%f", &b[i]);
    	}
    	fclose(mywFile);
	
	// Initialize  GPU Variables
	float *dev_a,*dev_b,*dev_c;

	// Allocate Memory in Device
	cudaMalloc((void**)&dev_a, rows*cols*sizeof(float));
	cudaMalloc((void**)&dev_b, cols*sizeof(float));
        cudaMalloc((void**)&dev_c, rows*sizeof(float));

	// Send Data to Device
	cudaMemcpy(dev_a,a,rows*cols*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b, cols*sizeof(float), cudaMemcpyHostToDevice);

	jobs = cols;
        BLOCKS = (jobs + THREADS - 1)/THREADS;	
	// Call GPU Function
	dotPro<<<BLOCKS, THREADS>>>(dev_a,dev_b,dev_c,rows,cols);
	
	// Send Result to Host Memory
	cudaMemcpy(c,dev_c,rows*sizeof(float), cudaMemcpyDeviceToHost);

	// Print Result &  Store  in a file
	for(int i = 0; i < rows; i++){
	printf("C[%d] = %f\t",i,c[i]);
	printf("\n");
	}	

	// Free Device Space
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

        return 0;	
}


