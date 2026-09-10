// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16* __restrict__ in,
                                   uint8_t* __restrict__ dataOut,
                                   uint8_t* __restrict__ sfOut, int M, int K) {
    int threadnum=gridDim.x*blockDim.x;
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    for(int i=idx;i<M*(K/16);i+=threadnum){
        int start=i*16;
        float maxn=fabsf(__bfloat162float(in[start]));
        for(int j=1;j<16;j++){
            float v=fabsf(__bfloat162float(in[start+j]));
            if(v>maxn) maxn=v;
        }
        __nv_fp8_e4m3 sf=__nv_fp8_e4m3(maxn/6.0f);
        float s=float(sf);
        float inv=s!=0.f ? 1.0f/s : 0.f;
        int row=i/(K/16);
        int group=i%(K/16);
        sfOut[sf_swizzled_offset(row,group,nvfp4_num_ktiles(K))]=sf.__x;
        for(int j=0;j<8;j++){
            // 两个 scaled FP4 合并为一个 byte，前一个元素放低 4 位。
            float lo=__bfloat162float(in[start+2*j])*inv;
            float hi=__bfloat162float(in[start+2*j+1])*inv;
            __nv_fp4x2_e2m1 packed(make_float2(lo,hi));
            dataOut[start/2+j]=packed.__x;
        }
    }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16* in, uint8_t* dataOut,
                               uint8_t* sfOut, int M, int K, int sms) {
    if (M <= 0 || K <= 0) return;
    constexpr int BLOCK = 128;
    const int64_t groups = int64_t(M) * (K / NVFP4_GROUP);
    const int64_t needed = (groups + BLOCK - 1) / BLOCK;
    const int limit = (sms > 0 ? sms : 1) * 8;
    const int grid = needed < limit ? int(needed) : limit;
    nvfp4_quant_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}
