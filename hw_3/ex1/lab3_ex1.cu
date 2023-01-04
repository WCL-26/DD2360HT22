
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define DataType double


__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx < len) {
      out[idx] = in1[idx] + in2[idx];
      }
}

//@@ Insert code to implement timer start
double Tstart() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
//@@ Insert code to implement timer stop
double duration(double startime) {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return (((double)tp.tv_sec + (double)tp.tv_usec*1.e-6) - startime);
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength=atoi(argv[1]);
  
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostInput2 = (DataType*)malloc(inputLength*sizeof(DataType));
  hostOutput = (DataType*)malloc(inputLength*sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

for (int i = 0; i < inputLength; ++i) {
     DataType random_1 = rand() / (DataType) RAND_MAX; 
     DataType random_2 = rand() / (DataType) RAND_MAX; 
     hostInput1[i] = random_1;
     hostInput2[i] = random_2;
    }
double start_time = Tstart();
for(int i=0;i<inputLength;i++){
   resultRef[i]=hostInput1[i]+hostInput2[i];
  }

 double h_duration = duration(start_time);
 //@@ Insert code below to allocate GPU memory here
 cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
 cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
 cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

 //@@ Insert code to below to Copy memory to the GPU here
 double start_1 = Tstart();
 cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
 cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
 cudaDeviceSynchronize();
 double duration_1 = duration(start_1);
 //@@ Initialize the 1D grid and block dimensions here
 int Db = 128;
 int Dg = (inputLength + Db - 1) / Db;

 //@@ Launch the GPU Kernel here
 double start_2 = Tstart();
 vecAdd<<<Dg,Db>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
 cudaDeviceSynchronize();
 double d_duration = duration(start_2);

 
  //@@ Copy the GPU memory back to the CPU here
  double start_3 = Tstart();
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double duration_2 = duration(start_3);

  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<inputLength;i++){
    if(resultRef[i] != hostOutput[i] && abs(resultRef[i]-hostOutput[i])>0.001 ){
        printf("Error counting numbers: %f",abs(resultRef[i]-hostOutput[i]) );
        return 0;
    }
  }
  printf("sum verified: Correct!\n");
  printf("Time Host->Device: %f - Time Device->Host: %f\n",duration_1,duration_2);
  printf("CPU time: %f - GPU time: %f\n",h_duration,d_duration);
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}