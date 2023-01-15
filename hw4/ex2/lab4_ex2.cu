
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
  int S_seg;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]);
  S_seg = atoi(argv[2]);
  printf("The input length is %d and the length of each segment is: %d\n", inputLength,S_seg);
  int N_seg  = ceil(inputLength / S_seg);
  printf("The number of segments is: %d\n", N_seg);
  //int B_seg = S_seg * sizeof(DataType);

  ///cudaStream_t stream[nStream];
  ///for(int i=0;i<nStream;++i)
    ///cudaStreamCreate(&stream[i]);



  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocHost(&hostInput1,inputLength * sizeof(DataType));
  cudaMallocHost(&hostInput2,inputLength * sizeof(DataType));
  cudaMallocHost(&hostOutput,inputLength * sizeof(DataType));


  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

for (int i = 0; i < inputLength; ++i) {
     DataType random_1 = rand() / (DataType) RAND_MAX; 
     DataType random_2 = rand() / (DataType) RAND_MAX; 
     hostInput1[i] = random_1;
     hostInput2[i] = random_2;
    }
///resultRef = (DataType*)malloc(inputLength*sizeof(DataType));
///for(int i=0;i<inputLength;i++){
   ///resultRef[i]=hostInput1[i]+hostInput2[i];
 /// }

 //double h_duration = duration(start_time);
 //@@ Insert code below to allocate GPU memory here
 cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
 cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
 cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

 // Create CUDA streams
 cudaStream_t stream[N_seg]; 
  // strat timer
 double start_1 = Tstart();

 for(int i = 0; i < N_seg; i++) {
    cudaStreamCreate(&stream[i]);
}

 //@@ Insert code to below to Copy memory to the GPU here
// double start_1 = Tstart();
 //cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
 //cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
 //cudaDeviceSynchronize();
 //double duration_1 = duration(start_1);
 //@@ Initialize the 1D grid and block dimensions here
 int Db = 128;
 int Dg = (inputLength + Db - 1) / Db;

 //@@ Launch the GPU Kernel here
 //double start_2 = Tstart();
 //vecAdd<<<Dg,Db>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
 //cudaDeviceSynchronize();
 //double d_duration = duration(start_2);
  for(int i = 0; i < N_seg; ++i){
    int offset= i * S_seg;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], S_seg * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], S_seg * sizeof(DataType), cudaMemcpyHostToDevice,stream[i]);
    vecAdd<<<Dg,Db,0,stream[i]>>>(&deviceInput1[offset],&deviceInput2[offset],&deviceOutput[offset],S_seg);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], S_seg * sizeof(DataType), cudaMemcpyDeviceToHost,stream[i]);
  }
  //@@ Copy the GPU memory back to the CPU here
  //double start_3 = Tstart();
  //cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
 // cudaDeviceSynchronize();
  //double duration_2 = duration(start_3);
    for(int i=0;i< N_seg;++i)
    cudaStreamDestroy(stream[i]);

    cudaDeviceSynchronize();
    double d_duration = duration(start_1);
    printf("The execution time is: %lf.\n", d_duration);
  //@@ Insert code below to compare the output with the reference
  for(int i=0;i<inputLength;i++){
    if(resultRef[i] != hostOutput[i] && abs(resultRef[i]-hostOutput[i])>0.001 ){
        printf("Error counting numbers: %lf",abs(resultRef[i]-hostOutput[i]) );
        return 0;
    }
  }
  printf("sum verified: Correct!\n");
  // printf("Time Host->Device: %f - Time Device->Host: %f\n",duration_1,duration_2);
  //printf("CPU time: %f - GPU time: %f\n",h_duration,d_duration);
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here

  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);
  return 0;
}