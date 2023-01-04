
#include <stdio.h>
#include <sys/time.h>
#include <random>


#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
   
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ unsigned int s_bins[NUM_BINS];

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    if (i < num_bins) {
      s_bins[i]=0;
    }
  } 
  __syncthreads();
  

  if (idx < num_elements) {
        atomicAdd(&(s_bins[input[idx]]), 1);
  }
   __syncthreads(); 


  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    if (i < num_bins) {
        atomicAdd(&(bins[i]), s_bins[i]);
      }
  }
    
}


__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
    int idx =  blockIdx.x * blockDim.x  + threadIdx.x;
    
    if (idx >= num_bins)
        return;

    if (bins[idx] > 127)
        bins[idx] = 127;

}


int main(int argc, char **argv) {
  
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    

  //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*)malloc(inputLength*sizeof(unsigned int));
    hostBins = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
    resultRef = (unsigned int*)malloc(NUM_BINS*sizeof(unsigned int));
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  //srand(clock());
    for (int i = 0; i < inputLength; ++i) 
    {
        hostInput[i] = rand() % NUM_BINS;
        printf("%d ",hostInput[i]);
    }
    //printf("\n\n");
    
  //@@ Insert code below to create reference result in CPU
    for (int i = 0; i < NUM_BINS; i++) 
        resultRef[i]=0;

    for (int i = 0; i < inputLength; i++) 
    {
        if(resultRef[hostInput[i]] < 127)
        {
            resultRef += 1;
        }
    }
  
  //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
    dim3 Dg_h(1024,1,1);
    dim3 Db_h(1024,1,1);

  //@@ Launch the GPU Kernel here
    histogram_kernel<<<Dg_h, Db_h>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    //cudaDeviceSynchronize();
  //@@ Initialize the second grid and block dimensions here
    dim3 Dg_d(1024,1,1);
    dim3 Db_d(1024,1,1);

  //@@ Launch the second GPU Kernel here
    convert_kernel<<<Dg_d, Db_d>>>(deviceBins, NUM_BINS);
   // cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
    int equality = 1;
    for (int i = 0; i < NUM_BINS; ++i) 
    {
       // printf("resultRef[%d] = %d\n", i, resultRef[i]);
        //printf("hostBins[%d] =  %d\n", i, hostBins[i]);       
        if (hostBins[i] != resultRef[i]) 
        {
            equality = 0;
            //break;
        }
    }
    if (equality == 1) {
        printf("CPU and GPU results are equal.\n");
    } 
    else 
    {
        printf("CPU and GPU results are NOT equal.\n");
    }
   // for(int i = 0;i < NUM_BINS;i++)
        //if(hostBins[i] != resultRef[i])
        //{
         //   printf("CPU and GPU results are NOT equal.\n");
        //    return 0;
       // }
   // printf("CPU and GPU results are equal.\n");

    FILE *fptr;
    fptr = fopen("./result.txt","w");
    if(fptr == NULL)
    {
      printf("Error!");   
      exit(1);             
    }
    for(int i = 0;i < NUM_BINS;i++){
      fprintf(fptr,"%d\n",resultRef[i]);
    }

       


  //@@ Free the GPU memory here
    
    cudaFree(deviceInput);
    cudaFree(deviceBins);

  //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}