/* pod-dev-timer-catlass 的 W4A8 Wave 任务图。可直接 file:// 打开，也可由 Node 测试。 */
(function (root) {
  "use strict";
  const SOURCE = "1bae0a346894e802677b3cccfff7b6f028e705da";
  const MGROUP_SOURCE = "e40f54884abd97d57604c679fe68baf300b4d5c9";

  function validate(c) {
    if (!["shared","dual_same","dual_stagger"].includes(c.catlassCursorPolicy ?? "shared")) throw new Error("CATLASS：未知游标策略，支持 shared / dual_same / dual_stagger");
    if (!["swizzle3","mgroup"].includes(c.catlassTileOrder ?? "swizzle3")) throw new Error("CATLASS：未知 Tile 排序，支持 swizzle3 / mgroup");
    for (const key of ["bs", "ep", "topk", "expertsPerRank", "cores", "hidden", "intermediate", "fmGroup"]) {
      if (!Number.isSafeInteger(c[key]) || c[key] < 1) throw new Error(`CATLASS：${key} 必须为正整数`);
    }
    if (c.topk > c.ep * c.expertsPerRank) throw new Error("CATLASS：TopK 超过总专家数");
    if (!Number.isSafeInteger(c.ep*c.bs*c.topk)) throw new Error("CATLASS：全局路由数量超出安全整数范围");
    for (const key of ["peakTflops", "gmm1Eff", "gmm2Eff", "commUtil", "fmPayload", "fmRawGbps", "closGbps"]) {
      if (!Number.isFinite(c[key]) || c[key]<=0) throw new Error(`CATLASS：${key} 必须为有限正数`);
    }
    if (c.commVariant !== "base") throw new Error("CATLASS 分支模式仅支持 Base 路由通信");
    if (c.tileM !== 256 || c.gmm1TileN !== 256 || c.gmm2TileN !== 256) {
      throw new Error("CATLASS 分支的调度 Tile 固定为 M=256、GMM1/GMM2 N=256；请恢复这些参数");
    }
    if (c.intermediate % 2) throw new Error("CATLASS：融合 N 必须为偶数（gate/up 各 N/2）");
    if (!Number.isInteger(c.catlassRank) || c.catlassRank < 0 || c.catlassRank >= c.ep) {
      throw new Error("CATLASS：观察 Rank 必须在 [0, EP) 内");
    }
    for (const key of ["catlassPrologue1", "catlassPrologue2", "swigluGap", "fixpipeGap"]) {
      if (!Number.isFinite(c[key]) || c[key] < 0) throw new Error(`CATLASS：${key} 必须为有限非负数`);
    }
    const groups = Math.ceil(c.bs * c.topk / 256) + Math.min(c.bs * c.topk, c.expertsPerRank);
    if (c.bs * c.topk > 200000 || groups * (Math.ceil(c.intermediate / 512) + Math.ceil(c.hidden / 256)) > 80000) {
      throw new Error("CATLASS：任务图过大，请减少 bs、TopK 或维度（最多20万路由、约8万计算 Tile）");
    }
  }

  // 反解 gen_balanced_expert_ids：i=(localExpert+q*L)*EP+rank。
  // 按源 rank 顺序排列接收行，数量是整数；不会把 bs*K/L 的小数作为 GEMM 的 M。
  function routes(c) {
    const rows = Array.from({length:c.expertsPerRank}, () => []);
    const total = c.ep * c.bs * c.topk;
    for (let e=0; e<c.expertsPerRank; e++) {
      for (let i=e*c.ep+c.catlassRank; i<total; i+=c.ep*c.expertsPerRank) {
        rows[e].push(Math.floor(i/(c.bs*c.topk)));
      }
    }
    return rows;
  }

  function swizzleTiles(stage, slice, c, offset) {
    const outputN = stage === "GMM1" ? c.intermediate/2 : c.hidden;
    const nTiles = Math.ceil(outputN/256), tasks=[];
    // opt-order 分支只将 Swizzle<3,0> 改为 <1,0>，奇数组 N 反向仍保留。
    // band 是当前专家 slice 内的编号，不使用全局 M-group 的奇偶性。
    const span=c.catlassTileOrder==="mgroup"?1:3;
    for (let t=0; t<slice.mGroups*nTiles; t++) {
      const block = Math.floor(t/(span*nTiles)), within=t%(span*nTiles);
      const valid = Math.min(slice.mGroups-block*span, span);
      const mi = block*span + within%valid;
      let ni = Math.floor(within/valid);
      if (block%2) ni=nTiles-1-ni;
      tasks.push({stage, wave:slice.wave, expert:slice.expert, tile:t,
        core:(offset+t)%c.cores, m:Math.min(256,slice.tokenCount-mi*256),
        n:Math.min(256,outputN-ni*256), nIndex:ni,
        mGroup:slice.tokenStart/256+mi, tokenStart:slice.tokenStart+mi*256});
    }
    return {tasks, offset:(offset+tasks.length)%c.cores};
  }

  function simulate(c, h) {
    validate(c);
    const rows=routes(c), counts=rows.map(x=>x.length), prefix=[0];
    counts.forEach(n=>prefix.push(prefix[prefix.length-1]+n));
    const mGroupsPerWave=c.waveMGroups || h.calcMGroupsPerWave(c);
    const waves=h.packDynamicWaves(counts,c,mGroupsPerWave);
    const cursorPolicy=c.catlassCursorPolicy??"shared";
    const firstWaveGmm1Tiles=waves[0].reduce((sum,s)=>sum+s.mGroups*Math.ceil(c.intermediate/512),0);
    // 仅初始化时错开；后续按各阶段真实 slice Tile 数延续，不硬编码核偏移或重置 Wave。
    const cursorStarts={GMM1:0,GMM2:cursorPolicy==="dual_stagger"?firstWaveGmm1Tiles%c.cores:0};
    const cursors={...cursorStarts},cursorAssignments=[];
    const nodes=[], tails=new Map(), dispatchGroups=new Map(), activationGroups=new Map();
    const g2Inputs=[];
    const key=(e,g)=>`${e}:${g}`;
    function node(spec, lane, deps=[], duration=0) {
      const n={...spec,id:nodes.length,lane,deps:[],next:[],duration};
      n.label=`${lane} / W${n.wave ?? "-"} ${n.stage} E${n.expert ?? "-"} T${n.tile ?? "-"}`;
      const before=tails.get(lane);
      n.deps=[...new Set([...deps, before].filter(x=>x!=null))];
      nodes.push(n); tails.set(lane,n.id); return n.id;
    }
    function append(map,k,id) { if(!map.has(k))map.set(k,[]); map.get(k).push(id); }
    function traffic(expert,start,end,width) {
      const peers=new Map();
      for(let r=start;r<end;r++) {
        const peer=rows[expert][r];
        peers.set(peer,(peers.get(peer)||0)+width);
      }
      return [...peers].map(([peer,bytes])=>({peer,bytes}));
    }
    const dispatchRowBytes=c.hidden+Math.ceil(c.hidden/32), combineRowBytes=2*c.hidden;
    // 首 Wave 一次按全局行区间分工，稳态按下一 Wave 的单专家 slice 分工。
    function dispatch(slices) {
      if(!slices.length)return;
      const begin=prefix[slices[0].expert]+slices[0].tokenStart;
      const last=slices[slices.length-1];
      const end=prefix[last.expert]+last.tokenStart+last.tokenCount;
      const total=end-begin, base=Math.floor(total/c.cores), rem=total%c.cores;
      for(let core=0;core<c.cores;core++) {
        const logical=(core-begin%c.cores+c.cores)%c.cores;
        const lo=begin+logical*base+Math.min(logical,rem), hi=lo+base+(logical<rem?1:0);
        for(const s of slices) {
          const from=Math.max(lo,prefix[s.expert]+s.tokenStart)-prefix[s.expert];
          const to=Math.min(hi,prefix[s.expert]+s.tokenStart+s.tokenCount)-prefix[s.expert];
          if(to<=from)continue;
          const id=node({stage:"Dispatch",wave:s.wave,expert:s.expert,core,tile:-1,
            tokenStart:from,m:to-from,n:c.hidden,commKind:"dispatch",rowBytes:dispatchRowBytes,
            peerBytes:traffic(s.expert,from,to,dispatchRowBytes)},`AIV1 ${core}`);
          for(let g=Math.floor(from/256);g<Math.ceil(to/256);g++)append(dispatchGroups,key(s.expert,g),id);
        }
      }
    }
    let offset=0, startBlockAfterFirstGmm1=0;
    function compute(stage,slice) {
      const start=cursorPolicy==="shared"?offset:cursors[stage];
      const assigned=swizzleTiles(stage,slice,c,start);
      if(cursorPolicy==="shared")offset=assigned.offset;
      else cursors[stage]=assigned.offset;
      cursorAssignments.push({wave:slice.wave,expert:slice.expert,tokenStart:slice.tokenStart,stage,start,end:assigned.offset,tileCount:assigned.tasks.length});
      for(const spec of assigned.tasks) {
        spec.cursorStart=start;
        const lane=`AIC ${spec.core}`;
        // 只建模显式输入的“未被 Tile 时间吸收的供数延迟”。采用 Tile 级交接，
        // 不把该参数当成源码的 K-block 双缓冲/MTE2 吞吐精确模型。
        const supply=c[stage==="GMM1"?"catlassPrologue1":"catlassPrologue2"];
        const p=node({...spec,stage:`${stage} 权重供数`},`AIV0 ${spec.core}`,[tails.get(lane)],supply);
        const deps=[p];
        if(stage==="GMM1")deps.push(...(dispatchGroups.get(key(spec.expert,spec.mGroup))||[]));
        const a=node(spec,lane,deps,h.tileDuration(stage,spec.m,spec.n,c));
        if(stage==="GMM1") {
          const act=node({...spec,stage:"SwiGLU+Quant"},`AIV1 ${spec.core}`,[a],c.swigluGap);
          append(activationGroups,key(spec.expert,spec.mGroup),act);
        } else {
          g2Inputs.push({id:a,key:key(spec.expert,spec.mGroup)});
          const fix=node({...spec,stage:"Fixpipe"},lane,[a],c.fixpipeGap);
          node({...spec,stage:"Combine",commKind:"combine",rowBytes:combineRowBytes,
            peerBytes:traffic(spec.expert,spec.tokenStart,spec.tokenStart+spec.m,spec.n*2)},
            `AIV1 ${spec.core}`,[fix]);
        }
      }
    }
    dispatch(waves[0]);
    waves.forEach((slices,w)=>{
      const next=waves[w+1]||[];
      for(let i=0;i<Math.max(slices.length,next.length);i++) {
        if(next[i])dispatch([next[i]]);
        if(slices[i])compute("GMM1",slices[i]);
      }
      if(w===0)startBlockAfterFirstGmm1=cursorPolicy==="shared"?offset:cursors.GMM1;
      slices.forEach(s=>compute("GMM2",s));
    });
    for(const input of g2Inputs) {
      const deps=activationGroups.get(input.key);
      if(!deps?.length)throw new Error(`CATLASS：GMM2 ${input.key} 缺少 Activation 生产者`);
      nodes[input.id].deps=[...new Set([...nodes[input.id].deps,...deps])];
    }

    const metadataBytesPeer=Math.ceil(c.ep*c.expertsPerRank/128)*128*4;
    const metaTiming=h.networkTiming(metadataBytesPeer,c,c.networkStartup);
    // 双工近似：Dispatch 接收与 Combine 发送各有资源；每个 FM peer 独占，Clos 同向共享。
    const net={dispatch:new Map(),combine:new Map()}, services={dispatch:new Map(),combine:new Map()};
    const fmGroup=c.topologyPreset==="fullmesh"?c.ep:c.fmGroup;
    function network(n,release) {
      const demands=new Map(), fmBw=c.fmRawGbps*c.fmPayload*c.commUtil, closBw=c.closGbps*c.commUtil;
      let fmBytes=0,closBytes=0,localBytes=0;
      for(const {peer,bytes} of n.peerBytes) {
        if(peer===c.catlassRank){localBytes+=bytes;continue;}
        const fm=Math.floor(peer/fmGroup)===Math.floor(c.catlassRank/fmGroup);
        const path=fm?`FM ${peer}`:"Clos";
        demands.set(path,(demands.get(path)||0)+bytes);
        if(fm)fmBytes+=bytes;else closBytes+=bytes;
      }
      const paths=[];
      for(const [resource,bytes] of demands) {
        const bandwidth=resource==="Clos"?closBw:fmBw;
        const startup=n.commKind==="dispatch"?c.networkStartup:0;
        const serviceUs=bytes/(bandwidth*1000)+startup;
        const start=Math.max(release,net[n.commKind].get(resource)||0),end=start+serviceUs;
        net[n.commKind].set(resource,end);
        services[n.commKind].set(resource,(services[n.commKind].get(resource)||0)+serviceUs);
        paths.push({path:resource,start,end,serviceUs,bytes,bandwidth,startup});
      }
      return {paths,localBytes,fmBytes,closBytes,release,
        start:paths.length?Math.min(...paths.map(x=>x.start)):release,
        end:paths.length?Math.max(...paths.map(x=>x.end)):release};
    }
    const heap=new h.MinHeap((a,b)=>a.ready-b.ready||a.id-b.id);
    nodes.forEach(n=>{n.pending=n.deps.length;n.ready=metaTiming.duration;
      n.deps.forEach(id=>nodes[id].next.push(n.id));
      if(!n.pending)heap.push({id:n.id,ready:n.ready});
    });
    const segments=[{stage:"Metadata",core:-1,expert:-1,tile:-1,wave:-1,start:0,end:metaTiming.duration}];
    let processed=0;
    while(heap.length) {
      const item=heap.pop(),n=nodes[item.id];
      n.start=item.ready;n.end=item.ready+n.duration;
      if(n.commKind)Object.assign(n,network(n,item.ready));
      n.blockers=n.deps.map(id=>({kind:nodes[id].stage,label:nodes[id].label,ready:nodes[id].end}));
      if(n.commKind&&n.start>item.ready)n.blockers.push({kind:"network",label:"同向网络链路排队",ready:n.start});
      // 导出不含图边，避免结果/浏览器缓存膨胀。
      const {deps,next,pending,ready,...segment}=n;segments.push(segment);
      processed++;
      for(const id of n.next) {
        const target=nodes[id];target.ready=Math.max(target.ready,n.end);
        if(--target.pending===0)heap.push({id,ready:target.ready});
      }
    }
    if(processed!==nodes.length)throw new Error(`CATLASS：调度依赖未完成 ${processed}/${nodes.length}，请检查 case`);
    const stage=name=>segments.filter(s=>s.stage===name);
    const gmm1Segments=stage("GMM1"),gmm2Segments=stage("GMM2"),dispatchSegments=stage("Dispatch"),combineSegments=stage("Combine");
    // 掩盖量只统计链路实际活动，不能把通信任务包络里的排队/空隙算成服务。
    const dispatchActive=dispatchSegments.flatMap(s=>s.paths),combineActive=combineSegments.flatMap(s=>s.paths);
    const cube=[...gmm1Segments,...gmm2Segments];
    const min=list=>list.reduce((a,s)=>Math.min(a,s.start),Infinity);
    const max=list=>list.reduce((a,s)=>Math.max(a,s.end),0);
    const envelope=list=>list.length?max(list)-min(list):0;
    const total=max(segments),cubeStart=min(cube),cubeEnd=max(cube);
    const activeCoreUs=cube.reduce((a,s)=>a+s.end-s.start,0);
    const waitSegments=h.deriveWaitSegments(cube,c.cores);
    const coreLoad=Array.from({length:c.cores},(_,core)=>({core,gmm1Count:0,gmm2Count:0,gmm1Us:0,gmm2Us:0,computeUs:0,waitUs:0}));
    for(const s of cube){const load=coreLoad[s.core],prefix=s.stage==="GMM1"?"gmm1":"gmm2";load[prefix+"Count"]++;load[prefix+"Us"]+=s.end-s.start;load.computeUs+=s.end-s.start;}
    for(const s of waitSegments)coreLoad[s.core].waitUs+=s.end-s.start;
    const service=kind=>Math.max(0,...services[kind].values());
    const g={fmPeers:Math.min(fmGroup,c.ep-Math.floor(c.catlassRank/fmGroup)*fmGroup)-1,closPeers:0,uneven:false};
    g.closPeers=c.ep-g.fmPeers-1;
    return {c,segments,gmm1Segments,gmm2Segments,dispatchSegments,combineSegments,dispatchActive,combineActive,
      waitSegments,total,cubeStart,cubeEnd,activeCoreUs,coreLoad,
      cursorPolicy,cursorStarts,firstWaveGmm1Tiles,cursorAssignments,
      cursorFinal:cursorPolicy==="shared"?{shared:offset}:{...cursors},experimentalScheduler:cursorPolicy!=="shared",
      activePct:100*activeCoreUs/(c.cores*total),windowPct:100*activeCoreUs/(c.cores*(cubeEnd-cubeStart)),
      mfuPct:100*3*c.bs*c.topk*c.hidden*c.intermediate/(c.peakTflops*1e6*total),
      sourceCommit:c.catlassTileOrder==="mgroup"?MGROUP_SOURCE:SOURCE,
      catlassTileOrder:c.catlassTileOrder??"swizzle3",expertCounts:counts,routing:"rank-major-balanced",schedulerMode:"catlass",
      tokensPerExpert:c.bs*c.topk/c.expertsPerRank,totalExperts:c.ep*c.expertsPerRank,
      mGroupsPerWave,waveCount:waves.length,waves,startBlockAfterFirstGmm1,finalStartBlock:cursorPolicy==="shared"?offset:null,
      waveEnvelopes:waves.map((slices,wave)=>{const ss=cube.filter(s=>s.wave===wave);return {wave,slices,start:min(ss),end:max(ss)};}),
      releaseMode:"配对 AIV1 / GM Tile",metaTiming,metadataBytesPeer,
      dispatchTiming:{duration:service("dispatch")/c.expertsPerRank},combineTiming:{duration:service("combine")/c.expertsPerRank},
      perExpertRowsPerPeer:c.bs*c.topk/(c.ep*c.expertsPerRank),
      dispatchBytesPeer:c.bs*c.topk/(c.ep*c.expertsPerRank)*dispatchRowBytes,
      combineBytesPeer:c.bs*c.topk/(c.ep*c.expertsPerRank)*combineRowBytes,
      tileCounts:{gmm1:gmm1Segments.filter(s=>s.expert===0).length,gmm2:gmm2Segments.filter(s=>s.expert===0).length},group:g,
      stage:{meta:metaTiming.duration,dispatch:service("dispatch"),gmm1:envelope(gmm1Segments),gmm2:envelope(gmm2Segments),
        combineService:service("combine"),combineEnvelope:envelope(combineSegments)},
      overlap:{dispatch:h.overlapDuration(dispatchActive,gmm1Segments),combine:h.overlapDuration(cube,combineActive),
        gmm12:h.overlapDuration(gmm1Segments,gmm2Segments)}};
  }
  const api={SOURCE,MGROUP_SOURCE,validate,routes,swizzleTiles,simulate};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  else root.CatlassWave=api;
})(globalThis);
