//m16n8k32 e4m3 mma、f32
#include <cuda_fp8.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <random>

//A:i=0~15;B:i=0~7
__device__ static int a_row_of(int lane, int i) {  return lane/4+(i/4)%2*8; }
__device__ static int a_col_of(int lane, int i) {  return (lane%4)*4+i%4+(i/8)*16; }
__device__ static int b_row_of(int lane, int i) {  return (lane%4)*4+i%4+(i/4)%2*16; }  // k
__device__ static int b_col_of(int lane, int i) {  return lane/4+(i/8)*4; }  // n
__device__ static int d_row_of(int lane, int i) {  return lane/4+(i/2)*8; }
__device__ static int d_col_of(int lane, int i) {  return (lane%4)*2+i%2; }

__global__ void ld_m(const uint8_t * sa,const uint8_t* sb,float* sd){
    int lane=threadIdx.x;
    uint32_t ra[4]; 
    for(int i=0;i<4;i++){
        uint8_t a[4];
        for(int j=0;j<4;j++){
            a[j]=sa[a_row_of(lane,i*4+j)*32+a_col_of(lane ,i*4+j)];
        }
        ra[i]=a[3]<<24|a[2]<<16|a[1]<<8|a[0];
    }
    uint32_t rb[2];
    for(int i=0;i<2;i++){
        uint8_t b[4];
        for(int j=0;j<4;j++){
            b[j]=sb[b_row_of(lane,i*4+j)*8+b_col_of(lane,i*4+j)];
        }
        rb[i]=b[3]<<24|b[2]<<16|b[1]<<8|b[0];
    }
    float rc[4]={0,0,0,0};
    float rd[4]={0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(rd[0]), "=f"(rd[1]), "=f"(rd[2]), "=f"(rd[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1])
        ,"f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3])
    );
    for(int i=0;i<4;i++){
       sd[d_row_of(lane,i)*8+d_col_of(lane,i)]=rd[i]; 
    }
}

int main(int argc, char** argv) {
    const int M=16,N=8,K=32;
    const int size=sizeof(uint8_t);
    char* end = nullptr;
    unsigned long parsed = std::strtoul(argv[1], &end, 10);

    unsigned seed = static_cast<unsigned>(parsed);
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> dist(-8,7);
    uint8_t *ha = new uint8_t[16*32];
    uint8_t *hb = new uint8_t[32*8];
    float *fa = new float[16*32];
    float *fb = new float[32*8];
    for(int i=0;i<16*32;i++){
        __nv_fp8_e4m3 value(static_cast<float>(dist(rng)));
        ha[i]=value.__x;
        fa[i]=static_cast<float>(value);
    }
    for(int i=0;i<32*8;i++){
        __nv_fp8_e4m3 value(static_cast<float>(dist(rng)));
        hb[i]=value.__x;
        fb[i]=static_cast<float>(value);
    }
    uint8_t *da,*db;
    cudaMalloc(&da, 16*32*sizeof(uint8_t));
    cudaMalloc(&db, 32*8*sizeof(uint8_t));
    cudaMemcpy(da,ha,16*32*sizeof(uint8_t),cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,32*8*sizeof(uint8_t),cudaMemcpyHostToDevice);
    float *dd;
    cudaMalloc(&dd, 16*8*sizeof(float));
    ld_m<<<1,32>>>(da,db,dd);
    float *hd=new float[16*8];
    cudaMemcpy(hd,dd,16*8*sizeof(float),cudaMemcpyDeviceToHost);
    float *ref=new float[16*8];
    for (int i=0;i<M;i++){
        for (int n=0;n<N;n++){
            float sum=0;
            for (int k=0;k<K;k++){
                sum+=fa[i*K+k]*fb[k*N+n];
            }
            ref[i*N+n]=sum;
        }
    }
    for (int i=0;i<M;i++){
        for (int n=0;n<N;n++){
            if (hd[i*N+n]!=ref[i*N+n]){
            printf("MISMATCH");
            return 1;
            }
        }
    }
    printf("PASS");
    return 0;
}
