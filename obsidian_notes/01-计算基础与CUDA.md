<!-- auto-toc -->
<!-- auto-toc:start -->
- [[#计算基础|计算基础]]
  - [[#并行计算|并行计算]]
  - [[#任务分解方式|任务分解方式]]
  - [[#硬件与编程模型|硬件与编程模型]]
- [[#CUDA编程模型|CUDA编程模型]]
  - [[#why GPU|why GPU]]
  - [[#what's CUDA|what's CUDA]]
  - [[#内存层次|内存层次]]
  - [[#warp与occupancy|warp与occupancy]]
  - [[#CUDA API要点|CUDA API要点]]
<!-- auto-toc:end -->

---

# 计算基础
## 并行计算

- 程序是处理器执行的指令序列,经典流水包括取指、译码、执行、访存、写回；IPC 衡量每周期完成的指令数。
- 性能有两个不同目标:**延迟(latency)** 追求单条指令更快完成(提高频率),**吞吐量(throughput)** 追求单位时间内完成更多指令(核心数)。而==并行计算==能提升吞吐量,但对延迟的影响有限。
- 并行有三种层级:==指令级并行(ILP)、数据级并行(DLP)、线程级并行(TLP)==。
	- **ILP**:让彼此无数据依赖的指令在时间上重叠执行。现代CPU常借助流水线、超标量和乱序执行提取ILP；NVIDIA GPU通常不采用CPU式的宽动态乱序执行,而是依靠编译器安排独立指令,并由硬件从就绪的warp中选择指令发射。ILP对GPU仍然重要,例如循环展开可以增加ILP,较高ILP有时能在较低occupancy下隐藏延迟。。
	- **DLP**:同一操作作用在多个数据元素上,经典例子是SIMD和SIMT。GPU的核心设计思想正是大规模DLP,将数据并行映射到大量处理单元上。
	- **TLP**:多个线程或进程同时执行,每个线程处理独立的任务,通过线程调度与同步来协调。
	- **在GPU上**:每个线程执行的是同一段代码(即kernel),但处理的数据不同 → 本质上是DLP的表现;同时GPU通过数千个硬件线程(warps)轮转执行来隐藏访存延迟 → 这又是TLP的本质。因此GPU的并行架构实际上是DLP + TLP 的组合——通过数据并行来驱动,用线程并行来实现。
- 并行计算的重点是**并行化**和**并行度**
	- **并行化:** 指的是将原本串行代码改写为可并行执行的代码。难点在于识别可并行任务、处理数据依赖和通信机制、完成负载均衡。
	- **并行度**:指同时运行的线程/任务数量,一般和硬件资源相关(线程数、SM数等)。并行度并不等于加速比;实际加速比还受Amdahl定律和通信开销影响。
	- **能并行计算的场合**:大规模计算、无数据依赖/独立任务、相同操作在不同数据上。(*适合人工智能捏*)

## 任务分解方式
- **Domain decomposition**：切分数据域，各计算单元执行相似操作。
- **Functional decomposition**：按不同功能或阶段拆任务。
- **Pipeline parallelism**：把工作拆成连续阶段,让不同输入同时处于不同阶段；它通常可以看作functional decomposition的一种执行方式。[^1]

[^1]: 这一段写的有点抽象,详细点讲就是将流水线进行拆分,同时进行不同/相同任务的不同阶段,提高并行化;而ILP则是执行彼此并行的不同指令,提升处理器的利用率和吞吐率。和分解(decomposition)相比,更强调了并行处理


## 硬件与编程模型
- Flynn分类法将计算机架构分为四种:==SISD(单指令单数据)、SIMD(单指令多数据)、MISD(多指令单数据)、MIMD(多指令多数据)==[^2]。SIMD 强调在多个数据上同时执行同一指令(ex: AVX向量运算),MIMD 是多个处理器独立执行不同指令。现代CPU和GPU通常具有MIMD特征,在局部上有SIMD和SIMT
	- GPU:整体上是MIMD(不同SM可以跑不同的kernel),但单个SM内部是SIMT,不是SIMD[^3]。

[^2]: S(ingle),T(hread),M(ultiple),D(ata) 

# CUDA编程模型
## why GPU
- 少量控制,大量计算
- 高吞吐,部分指令高延迟,但是会通过驻留warp来隐藏延迟
- 通常具有较高的聚合内存带宽

## what's CUDA
| 层次          | 含义                  | 关键约束                          |
| ----------- | ------------------- | ----------------------------- |
| Thread      | 程序执行基本单元            | 私有寄存器和局部状态                    |
| Warp        | 32 个 thread 的硬件执行组  | 同周期通常执行同一指令                   |
| Block / CTA | 线程协作范围              | 可用 shared memory、barrier、原子操作 |
| Grid        | 一次 kernel 的全部 block | block 独立，调度顺序未定义              |
| SM          | 执行 block/warp 的硬件   | 寄存器、SMEM、warp 槽位等资源有限         |
| Cluster     | CC 9.0+ 的 block 集合  | 同一 GPC，可跨 block 通信            |
![[CPU&GPU.png]]
- 编译流程
![[compile.png]]

[^3]: 数据是被动的，线程是主动的。从计算能力7.0开始,GPU的线程有了独立的progarm counter,可以各自跳转.但是同一个warp内指令流还是统一的,如有分支会mask掉一部分线程(warp divergence),因此尽量避免warp内分支才是高效的。

## 内存层次
![[memory.png]]

- **速度方面**(一般情况):寄存器 > 共享内存 > constant缓存(只读)>全局内存。
- **Constant memory**是只读且带缓存的地址空间。同一warp只读取少数不同地址时效率较高;所有线程读取同一地址时可以广播,速度可能接近寄存器访问,而读取多个不同地址会被串行化。
- **Local memory**仅对所属线程可见,但物理上位于device memory。除了寄存器spill,大型自动数组或结构体、动态索引数组也可能被编译器放入local memory。其访问可能命中缓存,但通常比寄存器昂贵.


## warp与occupancy
- **Occupancy**:每SM中活跃的 warp 数 / 最大可容纳 warp 数。
- Global memory的延迟很大,所以GPU要通过**不断切换warp来隐藏延迟:** 当某个warp等待访存(**stall**),SM可以切换到另一个warp执行。
- 高 occupancy 常有助于隐藏延迟，但不是最终目标；过度追求 occupancy 可能牺牲寄存器复用和 ILP。

## CUDA API要点
- 所有 CUDA API 和 kernel launch 都应做错误检查；事件适合 GPU 侧计时。
- Stream是按序执行的操作序列:同一stream内的操作保持顺序,不同stream之间只有在依赖关系、硬件能力和资源允许时才可能重叠,并发并不保证发生;还应留意legacy default stream与per-thread default stream的不同语义。
- `cudaMallocManaged` 提供统一内存，简化编程但不消除数据迁移成本。
- 多卡需要显式处理设备、数据分布、通信和同步。