// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <random>
#include <vector>
#include "../common.h"

#ifndef STAGES
#define STAGES 3
#endif

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
}

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(mbar), "r"(phase));
    return done;
}

__global__ void gemm_pipeline(const __nv_bfloat16* gA, const __nv_bfloat16* gB,
                              float* gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem =
        (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    
    // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
    // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
    //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    __nv_bfloat16* smemA[NSTAGE];
    __nv_bfloat16* smemB[NSTAGE];
    uint32_t saaddr[NSTAGE], sbaddr[NSTAGE];
    __shared__ __align__(8) uint64_t full[NSTAGE], empty[NSTAGE];
    __shared__ uint32_t tmemaddr[1];
    uint32_t dst=(uint32_t)__cvta_generic_to_shared(tmemaddr);
    uint32_t fbaraddr[NSTAGE], ebaraddr[NSTAGE];
    int tileM=BM*blockIdx.x, tileN=BN*blockIdx.y;
    for(int i=0;i<NSTAGE;i++) {
        smemA[i] = (__nv_bfloat16*)&smem[i * (BM + BN) * BK * 2];
        smemB[i] = (__nv_bfloat16*)&smem[i *(BM + BN) * BK * 2 + BM * BK * 2];
        saaddr[i] = (uint32_t)__cvta_generic_to_shared(smemA[i]);
        sbaddr[i] = (uint32_t)__cvta_generic_to_shared(smemB[i]);
        fbaraddr[i] =(uint32_t)__cvta_generic_to_shared(&full[i]);
        ebaraddr[i] =(uint32_t)__cvta_generic_to_shared(&empty[i]);
        if(tid==0){
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(fbaraddr[i]), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(ebaraddr[i]), "r"(1));  
        }

    }
    __syncthreads();

    if (warp==0){

    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
        " [%0], %1;"
        :
        : "r"(dst), "r"(64)
        
    );
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");

}
    // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
    //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
    int iters=K / BK;
    __syncthreads();
    int prestage=min(NSTAGE,iters);
    __shared__ int tmaiter;
        
    if(tid<prestage){
        int tileK = tid * BK;
        int expected_bytes = (BM + BN) * BK * 2;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :
            : "r"(fbaraddr[tid]), "r"(expected_bytes)
            : "memory"
        );
        //这里tma搬运
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(saaddr[tid]),
            "l"(&tmapA),
            "r"(tileK),
            "r"(tileM),
            "r"(fbaraddr[tid])
            : "memory"
        );
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global"
            ".mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(sbaddr[tid]),
            "l"(&tmapB),
            "r"(tileK),
            "r"(tileN),
            "r"(fbaraddr[tid])
            : "memory"
        );
    }
    if(tid==0)tmaiter=prestage;
    __syncthreads();
    
    // (3) 主循环 it:
    //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
    //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
    //       的轮次算,第一次复用等的是上一轮使用的完成)
    //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
    //       继续发,失败立刻停,不许阻塞
    //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
    //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
    //       empty[it%NSTAGE]
    for (int it=0;it<iters;it++){
        int stage=it%NSTAGE;
        int repeat=it/NSTAGE&1;
        int tileK=it*BK;
        if(tmaiter<=it){
            __syncthreads();
            mbar_wait(ebaraddr[stage],repeat^1);
            if(tid==0){
                int expected_bytes=(BM+BN)*BK*2;
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(fbaraddr[stage]), "r"(expected_bytes)
                    : "memory"
                );
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];"
                    :
                    : "r"(saaddr[stage]),
                    "l"(&tmapA),
                    "r"(tileK),
                    "r"(tileM),
                    "r"(fbaraddr[stage])
                    : "memory"
                );
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];"
                    :
                    : "r"(sbaddr[stage]),
                    "l"(&tmapB),
                    "r"(tileK),
                    "r"(tileN),
                    "r"(fbaraddr[stage])
                    : "memory"
                );
                tmaiter++;
            }
            __syncthreads();
            
        }
        if((it+1<iters)&&tmaiter<=it+1){
            __syncthreads();
            if(tid==0&&mbar_try(ebaraddr[(it+1)%NSTAGE],((it+1)/NSTAGE&1)^1)){
                int expected_bytes=(BM+BN)*BK*2;
                int next_stage=(it+1)%NSTAGE;
                    asm volatile(
                        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                        :
                        : "r"(fbaraddr[next_stage]), "r"(expected_bytes)
                        : "memory"
                    );
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global"
                        ".mbarrier::complete_tx::bytes "
                        "[%0], [%1, {%2, %3}], [%4];"
                        :
                        : "r"(saaddr[next_stage]),
                        "l"(&tmapA),
                        "r"(tileK+BK),
                        "r"(tileM),
                        "r"(fbaraddr[next_stage])
                        : "memory"
                    );
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global"
                        ".mbarrier::complete_tx::bytes "
                        "[%0], [%1, {%2, %3}], [%4];"
                        :
                        : "r"(sbaddr[next_stage]),
                        "l"(&tmapB),
                        "r"(tileK+BK),
                        "r"(tileN),
                        "r"(fbaraddr[next_stage])
                        : "memory"
                    );
                    tmaiter++;
            }
            __syncthreads();   
            }
            
            mbar_wait(fbaraddr[stage], repeat);
            __syncthreads();
            uint32_t idesc=(1u << 4) |   (1u << 7) | (1u << 10) |     ((BN >> 3) << 17) |  ((BM >> 4) << 24); 
            if(tid==0){
                asm volatile("tcgen05.fence::after_thread_sync;");
                for (int round=0;round<4;round++){
                    uint32_t taddr=tmemaddr[0];
                    uint64_t adesc=make_desc_sm100(saaddr[stage]+round*32,0,1024,2);
                    uint64_t bdesc=make_desc_sm100(sbaddr[stage]+round*32,0,1024,2);
                    asm volatile(
                    "{\n"
                    "  .reg .pred accumulate;\n"
                    "  setp.ne.b32 accumulate, %4, 0;\n"
                    "  tcgen05.mma.cta_group::1.kind::f16 "
                    "    [%0], %1, %2, %3, accumulate;\n"
                    "}\n"
                    :
                    : "r"(taddr),
                    "l"(adesc),
                    "l"(bdesc),
                    "r"(idesc),
                    "r"(round+it)
                    : "memory"
                );
                }

                asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one"
                ".shared::cluster.b64 [%0];"
                :
                : "r"(ebaraddr[stage])
                : "memory"
                ); 
            }
        
    }
    // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue
    mbar_wait(ebaraddr[(iters-1)%NSTAGE], ((iters-1)/NSTAGE)&1);
    asm volatile("tcgen05.fence::after_thread_sync;");
    float t[8];
    for(int n=0;n<BN;n+=8){
        uint32_t addr=tmemaddr[0]+((warp*32)<<16)+n;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32"
            "{%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
            :"=f"(t[0]),"=f"(t[1]),"=f"(t[2]),"=f"(t[3]),"=f"(t[4]),"=f"(t[5]),"=f"(t[6]),"=f"(t[7])
            :"r"(addr)
        );
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for(int i=0;i<8;i++){
            gD[(tileM+tid)*N+tileN+i+n]=t[i];
        }
    }
    __syncthreads();
    if (warp == 0) 
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;":: "r"(tmemaddr[0]), "r"(64));
}

int main(int argc, char** argv) {
    int M = argc > 3 ? atoi(argv[1]) : 4096;
    int N = argc > 3 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (M % BM || N % BN || K % BK) {
        printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
        return 1;
    }
    size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-3, 3);
    std::vector<__nv_bfloat16> hA(nA), hB(nB);
    for (auto& v : hA) v = __float2bfloat16((float)dist(rng));
    for (auto& v : hB) v = __float2bfloat16((float)dist(rng));
    __nv_bfloat16 *dA, *dB;
    float *dD, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, nA * 2));
    CUDA_CHECK(cudaMalloc(&dB, nB * 2));
    CUDA_CHECK(cudaMalloc(&dD, nD * 4));
    CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

    // TODO:tensor map 从你的 4.2 原样复制。
    CUtensorMap tmapA = {}, tmapB = {};
    cuuint64_t adim[2]={K,M},bdim[2]={K,N};
    cuuint32_t aboxdim[2]={BK,BM},bboxdim[2]={BK,BN};
    cuuint32_t elem_strides[2]={1,1};
    cuuint64_t strides[1]={K*2};
    CUresult aresult=cuTensorMapEncodeTiled(
        &tmapA,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        dA,
        adim,
        strides,
        aboxdim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUresult bresult=cuTensorMapEncodeTiled(
        &tmapB,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        dB,
        bdim,
        strides,
        bboxdim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if(aresult!=CUDA_SUCCESS||bresult!=CUDA_SUCCESS){
        printf("CUresult error");
        return 1;
    }

    dim3 grid(M / BM, N / BN);
    // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
    size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smemBytes));
    auto launch = [&] {
        gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA,
                                                tmapB);
    };
    launch();
    CUDA_CHECK_KERNEL();

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF,
                 K, dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(nD), ref(nD);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < nD; i++) bad += got[i] != ref[i];

    int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
    float ms = time_avg_ms(launch, iters);
    double tflops = 2.0 * M * N * K / (ms * 1e9);
    float cub_ms = time_avg_ms(
        [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                         CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                         CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        },
        iters);
    double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
    printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
           "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
           NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops,
           cub_tflops, 100.0 * tflops / cub_tflops);
    cublasDestroy(h);
    return bad != 0;
}
