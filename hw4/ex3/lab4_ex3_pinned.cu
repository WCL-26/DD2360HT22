#include <stdio.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    if( col < numBColumns && row < numARows) 
    {
        double sum=0.0f;
        for(int k=0;k<numAColumns;++k){
            sum+= A[row*numAColumns + k] * B[k*numBColumns + col];
            }
            C[row*numBColumns+col]=sum;
          }
}

void multiply(DataType *A, DataType *B, DataType *C, int numARows,
            int numAColumns, int numBRows, int numBColumns) {
   for (int i = 0; i < numARows; ++i) {
       for (int j = 0; j < numBColumns; ++j) {
           C[i*numBColumns + j] = 0.0;
           for (int k = 0; k < numAColumns; ++k) {
               C[i*numBColumns + j] += A[i*numAColumns + k] * B[k*numBColumns + j];
            }
        }
    }
}



int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
   if (argc<4){
    printf("input length invalid\n");
    return 0;
  }
  
  numARows=atoi(argv[1]);
  numAColumns=atoi(argv[2]);

  numBRows=atoi(argv[3]); 
  numBColumns=atoi(argv[4]);
  //numCRows=atoi(argv[1]); 
  //numCColumns=atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  //hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  //hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  //hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  //resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  cudaHostAlloc(&hostA, numARows * numAColumns * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostB, numBRows * numBColumns * sizeof(DataType), cudaHostAllocDefault);
  cudaHostAlloc(&hostC, numCRows * numCColumns * sizeof(DataType), cudaHostAllocDefault);
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; ++i) {
      for (int j = 0; j < numAColumns; ++j) {
          DataType randomNumber = rand() / (DataType) RAND_MAX; // 
          hostA[i*numAColumns + j] = randomNumber;
        }
    }
  for (int i = 0; i < numBRows; ++i) {
      for (int j = 0; j < numBColumns; ++j) {
          DataType randomNumber = rand() / (DataType) RAND_MAX; // 
          hostB[i*numBColumns + j] = randomNumber;

        }
    }
  multiply(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
 

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 Dg(numCColumns,numCRows,1);
  dim3 Db(32,32,1);

  //@@ Launch the GPU Kernel here
  gemm<<<Dg,Db>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC,  numCRows * numCColumns *sizeof(DataType), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numCRows; ++i) {
      for (int j = 0; j < numCColumns; ++j) {
          if (fabs(hostC[i*numCColumns + j] - resultRef[i*numCColumns + j]) > 1e-4){
              printf("Position: [%d, %d], Difference: %f\n", i, j, fabs(hostC[i*numCColumns + j] - resultRef[i*numCColumns + j]));
              }
          
          
        }
    }
  

  //@@ Free the GP U memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  cudaFree(resultRef);


  return 0;
}