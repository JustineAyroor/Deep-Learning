// NAME : JUSTINE J AYROOR
// UCID : ja573
// Assignment 2

// import libraries
#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
// Kernel Function
void dotPro(float *a, float *b, float *c, int rows, int cols, int jobs){
	
	int i,j,stop;
        float sum = 0;
	int tid = omp_get_thread_num();

	if((tid+1)*jobs > rows) stop=rows;
        else stop = (tid+1)*jobs;

        printf("thread id=%d, start=%d, stop=%d jobs=%d\n", tid, tid*jobs, stop,jobs);
	for(j = tid*jobs; j < stop; j++){
		sum = 0;
		for(i = 0; i < cols; i++){
			sum += a[i*rows+j]*b[i]; 
		}	
		c[j] = sum;
	}
}

// Main Function
int  main(int argc, char* argv[]){
	//intialize Variables
	int rows,cols,nprocs=0;
	int i,j,jobs;
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        //CUDA_DEVICE = atoi(argv[3]);
	nprocs = atoi(argv[3]);
	if(nprocs == 0){
		printf("No of processors not specified while running program");
	}

        printf("Rows=%d Cols=%d nprocs = %d \n",rows,cols,nprocs);
	float *a[rows];
	for(i = 0; i<rows; i++){
		a[i]=(float*)malloc((float)cols*sizeof(float));
	}
	float *aT = (float*)malloc((float)cols*rows*sizeof(float));
	float *w = (float*)malloc((float)cols*sizeof(float));
    	// Read Data File & Store in Array
	FILE *myFile;
    	myFile = fopen(argv[4], "r");
    	for (i = 0; i < rows; i++){
		for(j = 0; j < cols; j++){
   			fscanf(myFile, "%f", &a[i][j]);
    		}
	}
    	for(i = 0; i < cols; i++){
		for(j = 0; j < rows; j++){
			aT[rows*i+j] = a[j][i];
		}
    	}	
    	fclose(myFile);
	
	// Read Vector File & Store in Array 
    	FILE *mywFile;
	mywFile = fopen(argv[5], "r");
    	for(i = 0; i < cols; i++){
		fscanf(mywFile, "%f", &w[i]);
    	}
    	fclose(mywFile);
	
	// Initialize  GPU Variables
	float *c = (float*)malloc((float)rows*sizeof(float)); 
	jobs = ((rows+nprocs-1)/nprocs);	
	// Call GPU Function
	#pragma omp parallel num_threads(nprocs)
	dotPro(aT,w,c,rows,cols,jobs);
	
	// Print Result &  Store  in a file
	for(i = 0; i < rows; i++){
	printf("C[%d] = %f\t",i,c[i]);
	printf("\n");
	}

        return 0;	
}


