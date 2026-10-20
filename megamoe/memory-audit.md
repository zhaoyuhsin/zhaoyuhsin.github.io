# decode-master MTE / UB 学习模型

入口：`memory.html`，保留同目录 `memory-model.js`、`memory-page.js`。无网络依赖。配置单独保存，不覆盖主流水配置；可导出带有效参数、源码版本与结果的 JSON。

## 参考快照与范围

本地 `decode-master`：`9afe86e730606c739b682686b397708f47b342de`。源码以所标注提交为参考。未修改内核、未拉取远端、未跑NPU。

默认 H7168、融合N6144、bs96、TopK6、EP64、6专家/Rank、32核组、865 TFLOPS。A8W4：W4 E2M1打包，A/W8 E4M3，E8M0 scale，FP32累加、BF16写回；无共享专家，无Combine量化，TopkWeightsPrefetch=0。

精确Tile分配限制为**均衡整数路由、每专家M≤256、H/N为256倍数、H≤8192**，此时每专家只有一个M-group，避免把多M-group的Swizzle/尾块与实际分配混为一谈。未满足直接报错，不偷偷换模型。Dispatch逐核行数为均衡工作量估计；不复现每Peer请求顺序。

## 代码核对

以下路径均相对 `kernels/mega_moe/`：

| 路径 / 函数 | 模型使用的事实 |
|---|---|
| `mega_moe_wave_a8w4.h` / `ProcessAiv1PriorityStages` | 非抢占的就绪SwiGLU→Dispatch→就绪Token Combine优先级；每次完成任务再判断 |
| 同文件 / `NextAivSlice`、`ProcessMoeExpertStages` | GMM1/GMM2独立游标跨slice延续；GMM2物理核反向；bs≥4096才滞后一波 |
| `common/mega_moe_constants.h` | GMM1逻辑N128、物理gate/up N256；GMM2 N256；M256容量；Scale为2B/64元素；metadata=8×4=32B，不能按 `INT32_PER_256B` 名称推断256B |
| `stage/mega_moe_gmm1_activation.h` / `Gmm1AicMmadTileA8W4`、`Gmm1Aiv1EpilogueTileA8W4` | AIC写GM，AIV1从GM读取两路BF16再激活；A8W4非prefetch也不是直接FIX→UB |
| `blaze/prologue/block_prologue_mx_fp8fp4.h` | AIV0 GM→UB W4；展开W8后UB→L1；W4四槽总64KiB，W8四槽总128KiB，私有自检248KiB；两半K串行，L1双缓冲跨核握手 |
| `blaze/gemm/block/block_mmad_mx_fp8fp4.h` | AIC读A/Scale；W8来自AIV0；MTE1到L0；FIX输出BF16；每N Tile重新搬A |
| `stage/mega_moe_token_dispatch.h` | 对齐Token+Scale一行GM→UB；写接收Token/Scale和32B元信息；Dispatch返回前drain |
| `stage/mega_moe_gmm2_combine.h` / `RunTokenCombineSlice` | AIV1六槽Token行环，H≤8192，等待slice的GMM2生产者完成；元信息200–248KiB视图，ready区域自184KiB起，返回前drain |
| `blaze/epilogue/block_epilogue_ub_layout.h` | 输入与中间/输出低地址复用；视图容量不等于同时有效载荷 |
| `mega_moe_arch35.h` / `ProcessInputPreparationStage`、`ProcessWave` | 输入量化、compact routes/count/reset、起止同步、最终Unpermute |

Dispatch容量复现 `csrc/ops/mega_moe.cpp` 的 `DispatchConfig`：初始route batch上限12288、2～6行槽，剩余预算允许时扩张route batch。平台UB tiling预算以256KiB布局假设展示，不是通用芯片规格。

## 字节公式

R=bs×TopK，M=R/每卡专家数，I=N/2。每个GMM1 tile包含gate128+up128，物理Q=256；GMM2 Q=256。归约K分别为H、I：

- AIV0：W4读 `KQ/2`；W8写L1 `KQ`；UB接口最少读写 `2(KQ/2+KQ)`。
- AIC：GM读 `MK+(M+Q)×2ceil(K/64)`；MTE1有效载荷下界 `MK+KQ+ScaleA+ScaleB`；FIX写 `2MQ`。
- Cube有用计算 `2MQK`，除以 `P_Cube/cores`；不用96/256再打一次效率折扣。
- Dispatch：GM读 `R[Align256(H)+Align32(ceil(H/32))]`，接收端写 `R(H+ScaleH+32)`。
- SwiGLU：GM读 `4RI`，写 `R(I+ScaleI)`。UB接口最少读写 `R(14I+2ScaleI)`，未计临时scale数组及重复向量访存。
- Combine：读 `R(2H+32)`，写 `2RH`，本地数据也计入MTE，不占网络。

每个阶段累计Tile字节，按实际小M双游标分核累加GMM/AIV0/SwiGLU需求。各引擎时间下界=`最忙核总字节/(单核GB/s×1000)`；它是服务量，不是包络，不包括依赖等待。

GM逻辑访问不等于HBM物理流量。一个可调统一miss比例只用于读HBM下界，`所有已列GM读×miss/(4000×1000)`；不改变MTE逻辑访问。输入量化及Unpermute最少载荷也计入Rank读写。对称均衡假设下，网络端点的源读、目标写以每Rank相同工作量核算，不额外重复添加网络字节。写带宽未知，不用4000GB/s读峰值替代。

## UB / 反压

分别展示AIV0和AIV1；不同时刻复用地址不能累加判溢出。AIV1 SwiGLU两路输入视图各128KiB；BS96、逻辑N128时实际输入48KiB、输入最高访问地址175.75KiB；中间/量化临时视图复用低区，视图末端202KiB。后两者不可误读为硬件最大可用空间。

Combine六槽行环84KiB；当前每专家96行在32核均分，每核一个slice3行，通常连6槽都填不满。更多缓冲不必然有收益。

局部行环实验遵守：`readStart=max(MTE2可用, 同槽前一行完成)`，`writeStart=max(readEnd,MTE3可用)`，`slotRelease=writeEnd+完成延迟`。发送完成延迟只占槽，不占MTE3引擎（实验假设）。所有输入行已就绪；不模拟GMM2 ready、AIV1抢占、路由及全Rank网络竞争。图不是实测重放。

## 不确定性与不改变主流水的原因

153GB/s单AIC读、4000GB/s Rank读及865TFLOPS是用户既有工程输入。AIV0/AIV1 MTE2、MTE3、MTE1、FIX、UB bandwidth均默认留空；不能自动补成153GB/s。UB读写只给最少接口流量，bank冲突、scale临时数组及向量指令效率未知。

页面仅给**部分资源必要下界**，没有新E2E/MFU承诺。control轮询可能重复读；cache/对齐/包头、Vector算力、全Rank动态公平带宽、输入/输出细分UB容量等尚未完整建模。原主流水仍使用经验Tile效率；不把本页搬运时间叠加进去，以免把同一供数成本计两遍。下一步用匹配分支与shape的分引擎微基准+timer等待细分校准，再把共享资源仲裁引入离散事件流水。

## 验证

`node test-memory-model.cjs`：默认字节核对、跨阶段字节守恒、分核总量、M变化与缓存敏感性、POD/PC16网络下界、行环依赖、无效输入。
