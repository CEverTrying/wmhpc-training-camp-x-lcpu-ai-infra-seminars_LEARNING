#include<stdio.h>
#include<cstdlib>
#define cuda_check(exp) do{\
    cudaError_t result=exp;\
    if(result!=cudaSuccess){\
        fprintf(stderr ,"CUDA erro %s at %s:%d:%s\n",cudaGetErrorName(result),__FILE__,__LINE__,cudaGetErrorString(result));\
        exit(1);\
    }\
}while(0)

#define cuda_kernel_check() \
do{\
    cuda_check(cudaGetLastError());\
    cuda_check(cudaDeviceSynchronize());\
}while(0)

__global__ void kernel(float*x,float*y,int n){
    int stride=gridDim.x*blockDim.x;
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    for(int i=idx;i<n;i+=stride){
        y[i]=x[i]*2.0f+y[i];
    }
}

int main(int argc,char **argv){
    int num=atoi(argv[1]);
    if(num==0){
        printf("SUM=0\n");
        return 0;
    }
    int bytes=num*sizeof(float);
    int thread=256;
    int block=(num+thread-1)/thread;
    float* hx;
    float* hy;
    float* hz;
    cuda_check((cudaMallocHost(&hx,bytes)));
    cuda_check(cudaMallocHost(&hy,bytes));
    cuda_check(cudaMallocHost(&hz,bytes));
    for(int i=0;i<num;i++){
        hx[i]=((i % 2048) - 1024) * 0.5f;
        hy[i]=(i % 1024) - 512;
    }
    float* dx;
    float* dy;
    cuda_check(cudaMalloc(&dx,bytes));
    cuda_check(cudaMalloc(&dy,bytes));
    cuda_check(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dy,hy,bytes,cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));
    cuda_check(cudaEventRecord(start)); 
    kernel<<<block,thread>>>(dx,dy,num);
    cuda_kernel_check();
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));
    cuda_check(cudaMemcpy(hz,dy,bytes,cudaMemcpyDeviceToHost));
    float ms;
    cuda_check(cudaEventElapsedTime(&ms,start,stop));
    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    double sum=0;
    for(int i =0;i<num;i++){
        sum+=hz[i];
    }
    printf("SUM=%.0f\n", sum);
}