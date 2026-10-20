/* decode-master 9afe86e: A8W4 BS96 资源账本；不是替代原调度器的 E2E 预测。 */
(function (root) {
  'use strict';
  const VERSION = 'decode-master@9afe86e730606c739b682686b397708f47b342de/memory-v1';
  const defaults = Object.freeze({bs:96, H:7168, N:6144, topk:6, E:64, experts:6, cores:32,
    peak:865, aicRead:153, hbm:4000, miss:100, aiv0Read:null, aiv1Read:null, aivWrite:null,
    ubToL1:null, mte1:null, fix:null, ub:null, fm:34.81, clos:140, netUtil:100,
    ringRead:null, ringWrite:null, ringLatency:0, ringSlots:6, ringRows:3});
  const ceil = Math.ceil, align = (n,a) => ceil(n/a)*a, scale = n => ceil(n/64)*2;
  const time = (bytes,bw) => bytes === 0 ? 0 : bw > 0 ? bytes/(bw*1000) : null;
  const sum = a => a.reduce((s,x)=>s+x,0), max = a => Math.max(...a);
  function validate(p) {
    for (const k of ['bs','H','N','topk','E','experts','cores','ringSlots','ringRows'])
      if (!Number.isSafeInteger(p[k]) || p[k]<=0) throw Error(k+' 必须为正整数');
    if (p.E>128 || p.experts>128 || p.cores>64 || p.bs>32768) throw Error('学习模型范围：E≤128、专家/Rank≤128、核组≤64、bs≤32768');
    if (p.H>8192 || p.H%256 || p.N>32768 || p.N%256) throw Error('decode-master 学习模型要求 H≤8192 且 H、融合 N 为256倍数');
    if (p.topk>p.E*p.experts) throw Error('TopK 不能超过总专家数');
    const M=p.bs*p.topk/p.experts;
    if (!Number.isInteger(M) || M>256) throw Error('本页精确分核范围：均衡整数路由、每专家 M=bs×TopK/专家数 ≤256；Prefill 多 M-group 请使用主流水页');
    if (p.E>8 && p.E%8) throw Error('本页拓扑要求 E≤8 或 E 是8的倍数');
    for (const k of ['peak','aicRead','hbm','fm','clos']) if (!(Number.isFinite(p[k])&&p[k]>0)) throw Error(k+' 必须大于0');
    for (const k of ['aiv0Read','aiv1Read','aivWrite','ubToL1','mte1','fix','ub','ringRead','ringWrite'])
      if (p[k]!==null && !(Number.isFinite(p[k])&&p[k]>0)) throw Error(k+' 请留空（待校准）或填写正带宽');
    if (!(p.miss>=0 && p.miss<=100 && p.netUtil>0 && p.netUtil<=100)) throw Error('HBM miss 为0～100%，网络利用率为(0,100]%');
    if (!(Number.isFinite(p.ringLatency)&&p.ringLatency>=0) || p.ringRows>256 || p.ringSlots>6) throw Error('行环实验要求行数≤256、槽数≤6、延迟≥0');
  }
  // 小 M 只有一个 M-group，N 遍历无 Swizzle 歧义。两游标跨 slice 延续；GMM2 反向映射物理核。
  function assignments(p) {
    const M=p.bs*p.topk/p.experts, I=p.N/2, n1=I/128, n2=p.H/256;
    const cores=Array.from({length:p.cores},(_,id)=>({id,g1:0,g2:0,rows:0}));
    let c1=0,c2=0;
    for(let e=0;e<p.experts;e++) {
      for(let t=0;t<n1;t++) cores[(c1+t)%p.cores].g1++;
      for(let t=0;t<n2;t++) cores[p.cores-1-(c2+t)%p.cores].g2++;
      // Combine 按专家累计行数轮转；Dispatch 账本用同等均衡工作量，具体远端 peer 顺序不推断。
      for(let r=0;r<M;r++) cores[(e*M+r)%p.cores].rows++;
      c1=(c1+n1)%p.cores;c2=(c2+n2)%p.cores;
    }
    return {M,I,n1,n2,cores};
  }
  function ring(rows,slots,readUs,writeUs,latency=0) {
    if (readUs===null || writeUs===null) return null;
    let rd=0,wr=0,blocked=0;
    const release=Array(slots).fill(0),events=[];
    for(let i=0;i<rows;i++) {
      const start=Math.max(rd,release[i%slots]);blocked+=start-rd;
      const end=start+readUs;rd=end;
      const ws=Math.max(end,wr),we=ws+writeUs;
      wr=we;release[i%slots]=we+latency;
      events.push({row:i,readStart:start,readEnd:end,writeStart:ws,writeEnd:we,release:we+latency});
    }
    return {events,blocked,total:Math.max(...release),serial:rows*(readUs+writeUs+latency)};
  }
  function model(input={}) {
    const p={...defaults,...input};validate(p);
    const a=assignments(p),{M,I,cores}=a,R=p.bs*p.topk,rows=[],tile=[];
    const add=(stage,role,engine,bytes,formula,perCore,note='')=>{
      const bw={aicRead:p.aicRead,aivRead:role==='AIV0'?p.aiv0Read:p.aiv1Read,aivWrite:p.aivWrite,ubToL1:p.ubToL1,mte1:p.mte1,fix:p.fix,ub:p.ub}[engine];
      const per=perCore || Array(p.cores).fill(bytes/p.cores);
      rows.push({stage,role,engine,bytes,formula,perCore:per,maxCore:max(per),average:bytes/p.cores,
        floorUs:time(max(per),bw),bw,note});
    };
    for(const [stage,count,K,Q] of [['GMM1','g1',p.H,256],['GMM2','g2',I,256]]) {
      const t={stage,activation:M*K,scales:(M+Q)*scale(K),packed:K*Q/2,expanded:K*Q,
        output:M*Q*2,flops:2*M*Q*K};
      const n=sum(cores.map(c=>c[count])), per=b=>cores.map(c=>c[count]*b);
      t.count=n;t.cubeUs=t.flops/(p.peak*1e6/p.cores);tile.push(t);
      add(stage,'AIV0','aivRead',n*t.packed,'Tile: K×物理N×0.5；每个 M-group 重新展开',per(t.packed));
      add(stage,'AIV0','ubToL1',n*t.expanded,'Tile: K×物理N×1；W4 → W8 后翻倍',per(t.expanded));
      add(stage,'AIV0','ub',n*2*(t.packed+t.expanded),'2×(W4+W8)：MTE2写、Vector读写、MTE3读',per(2*(t.packed+t.expanded)),'UB接口逻辑字节；不含额外布局/指令访问');
      add(stage,'AIC','aicRead',n*(t.activation+t.scales),'Tile: M×K + (M+物理N)×2ceil(K/64)',per(t.activation+t.scales),'A和Scale由AIC读；W4不在此重复计数；不同N Tile重新读A');
      add(stage,'AIC','mte1',n*(t.activation+t.expanded+t.scales),'Tile: M×K + K×物理N + ScaleA+ScaleB',per(t.activation+t.expanded+t.scales),'L1侧有效载荷下界；未计L0格式/对齐放大');
      add(stage,'AIC','fix',n*t.output,'Tile: M×物理N×2 (BF16)，L0C→GM',per(t.output),'FP32累加不代表写回4字节');
    }
    const record=align(p.H,256)+align(ceil(p.H/32),32),D=R*record,C=R*p.H*2;
    const owned=cores.map(c=>c.rows);
    add('Dispatch','AIV1','aivRead',D,'R×[Align256(H)+Align32(ceil(H/32))]',owned.map(r=>r*record),'含本地和远端GM读取；逐核是均衡工作量估计，不重现源Rank任务顺序');
    add('Dispatch','AIV1','aivWrite',R*(p.H+scale(p.H)+32),'R×(H+ScaleH+32B metadata)',owned.map(r=>r*(p.H+scale(p.H)+32)));
    add('Dispatch','AIV1','ub',D+R*(p.H+scale(p.H)+32),'GM→UB写 + UB→GM读（metadata标量写另计）',owned.map(r=>r*(record+p.H+scale(p.H)+32)),'UB访问下界');
    const swIn=R*I*4,swOut=R*(I+scale(I));
    add('SwiGLU/Quant','AIV1','aivRead',swIn,'R×I×2路×2B，从GM重读GMM1输出',cores.map(c=>c.g1*M*128*4));
    add('SwiGLU/Quant','AIV1','aivWrite',swOut,'R×(I+2ceil(I/64))',cores.map(c=>c.g1*M*(128+4)));
    add('SwiGLU/Quant','AIV1','ub',R*(14*I+2*scale(I)),'2×输入BF16 + 2×中间BF16 + 2×量化输出/Scale',cores.map(c=>c.g1*M*(14*128+8)),'最少读写；maxExp/reciprocalScale、重复扫描、bank冲突未计');
    add('Combine','AIV1','aivRead',C+R*32,'R×(2H+32B metadata)',owned.map(r=>r*(2*p.H+32)));
    add('Combine','AIV1','aivWrite',C,'R×2H，Token行发送BF16',owned.map(r=>r*2*p.H),'含本地写；远端还受网络出口限速');
    add('Combine','AIV1','ub',2*C+R*32,'数据行MTE2写+MTE3读，metadata搬入',owned.map(r=>r*(4*p.H+32)),'不含标量metadata读取和ready扫描');
    // 起止阶段单列 Rank 载荷，核内分块和 Vector scratch 未纳入精确逐核模型。
    const boundary={inputRead:p.bs*p.H*2,inputWrite:p.bs*record,unpermuteRead:C+p.bs*p.topk*4,unpermuteWrite:p.bs*p.H*2};
    const read=sum(rows.filter(r=>['aicRead','aivRead'].includes(r.engine)).map(r=>r.bytes))+boundary.inputRead+boundary.unpermuteRead;
    const write=sum(rows.filter(r=>['aivWrite','fix'].includes(r.engine)).map(r=>r.bytes))+boundary.inputWrite+boundary.unpermuteWrite;
    const resource=[];
    for(const role of ['AIC','AIV0','AIV1']) for(const engine of ['aicRead','aivRead','aivWrite','ubToL1','mte1','fix','ub']) {
      const matches=rows.filter(r=>r.role===role&&r.engine===engine);if(!matches.length)continue;
      const per=cores.map((_,i)=>sum(matches.map(r=>r.perCore[i])));
      resource.push({role,engine,bytes:sum(per),maxCore:max(per),floorUs:time(max(per),matches[0].bw)});
    }
    const network=bytes=>Math.max(p.E>1?bytes/p.E/(p.fm*p.netUtil/100*1000):0,
      Math.max(0,p.E-8)/p.E*bytes/(p.clos*p.netUtil/100*1000));
    const fixed=align(p.E*p.experts*4,32)+align(p.experts*4,32),routeAligned=align(p.bs,256);
    let routeItems=Math.min(routeAligned,12288);
    const dispatchSlots=Math.min(6,Math.max(2,Math.floor((256*1024-fixed-routeItems*4)/(record+32))));
    if(routeItems<p.bs) {
      const expanded=Math.min(routeAligned,Math.floor((256*1024-fixed-dispatchSlots*(record+32))/4/256)*256);
      routeItems=Math.max(routeItems,expanded);
    }
    const ub={aiv0Reserved:192*1024,aiv0Ceiling:248*1024,dispatchSlots,
      dispatchBytes:fixed+routeItems*4+dispatchSlots*(record+32),
      swInputReserved:256*1024,swPayload:M*128*4,swAddressEnd:128*1024+(M-1)*512+256,
      swScratchEnd:256*256*3+256*256/32*5,
      combineRing:6*align(2*p.H,32),combineMetaReserved:1536*32,
      combineMetaActual:Math.min(1536,ceil(M/p.cores))*32,combineScanStart:184*1024};
    const cube=sum(tile.map(t=>t.flops*t.count))/(p.peak*1e6),hbmRead=time(read*p.miss/100,p.hbm);
    const r=ring(p.ringRows,p.ringSlots,time(2*p.H,p.ringRead),time(2*p.H,p.ringWrite),p.ringLatency);
    return {version:VERSION,p,a,rows,tile,boundary,resource,ub,read,write,hbmRead,cube,
      dispatchNetwork:network(D),combineNetwork:network(C),ring:r,
      knownFloor:Math.max(cube,hbmRead,network(D),network(C),...resource.map(r=>r.floorUs||0)),
      incomplete:resource.some(r=>r.floorUs===null)};
  }
  const api={VERSION,defaults,model,ring,time};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.MegaMoeMemory=api;
})(globalThis);
