/* 实测时间 + Shape 推算字节；不依赖仿真调度器，不把 interval 后缀解释成 Tile 数。 */
(function(root){
  "use strict";
  const defaults={bs:4096,hidden:7168,topk:6,ep:64,experts:6,cores:32,
    dispatchBytes:1,combineBytes:2,scaleGroup:32,includeScale:false,scope:"all",fmGroup:8,unit:"us"};
  const stageOf=name=>String(name||"").match(/^(?:wave_(?:\d+|global)_)?(?:AIV1?_)?(DISPATCH|COMBINE)(?:_\d+)?$/i)?.[1].toLowerCase()||null;
  const key=e=>JSON.stringify([e.pid,e.tid]);
  const nonnegativeInt=v=>v!==null&&v!==""&&v!==undefined&&Number.isSafeInteger(Number(v))&&Number(v)>=0;
  function parse(text,filename="trace"){
    text=text.replace(/^\uFEFF/,"").trim();
    if(!text)throw new Error("流水文件为空");
    if(text.length>100*1024*1024)throw new Error("文件超过100 MiB，请先按单次运行、单Rank导出");
    let raw=[],format="Chrome Trace JSON",warnings=[];
    if(text.startsWith("MEGAMOE_TRACE_TEXT_V1")){
      format="MEGAMOE_TRACE_TEXT_V1";
      const rank=text.match(/^rank=(\d+)/m)?.[1];
      if(rank===undefined)throw new Error("文本流水缺少 rank= 标记");
      let lane=null;const seenGroups=new Set();
      for(const [i,line] of text.split(/\r?\n/).entries()){
        const section=line.match(/^\[group=(\d+) (AIC|AIV[01])\]$/);
        if(section){lane={group_id:Number(section[1]),core_type:section[2]==="AIC"?"AIC":"AIV",sub_id:section[2]==="AIV1"?1:0};if(section[2]==="AIV1")seenGroups.add(Number(section[1]));continue;}
        if(line.startsWith("WARNING:"))warnings.push(line);
        if(!/^[-\d]/.test(line))continue;
        const m=line.match(/^([-\d.eE+]+)\s+([-\d.eE+]+)\s+("(?:[^"\\]|\\.)*")(?:\s+(\{.*\}))?\s*$/);
        if(!m||!lane)throw new Error(`文本第${i+1}行格式无效或缺少分组`);
        let args,name;try{args=m[4]?JSON.parse(m[4]):{};name=JSON.parse(m[3]);}catch{throw new Error(`文本第${i+1}行 JSON 字段无效`);}
        raw.push({ph:"X",name,ts:Number(m[1]),dur:Number(m[2]),pid:`rank${rank}`,tid:`group_${lane.group_id}_${lane.core_type}_${lane.sub_id}`,args:{...args,...lane,rank:Number(rank)}});
      }
      const declared=text.match(/^rank=\d+\s+groups=([\d,]+)/m)?.[1].split(",").map(Number)||[];
      const missing=declared.filter(g=>!seenGroups.has(g));
      if(missing.length)warnings.push(`文本头声明的 group ${missing.join(",")} 未在正文找到 AIV1 分组；实际包含 ${[...seenGroups].join(",")}，可能是截取内容。`);
    }else{
      let obj;try{obj=JSON.parse(text);}catch{throw new Error("无法解析 JSON；请导入完整 Chrome Trace JSON 或 MEGAMOE_TRACE_TEXT_V1 文本");}
      raw=Array.isArray(obj)?obj:obj?.traceEvents;
      if(!Array.isArray(raw))throw new Error("JSON 必须是事件数组或包含 traceEvents 数组");
    }
    const threads=new Map(),processes=new Map();
    for(const e of raw){
      if(e?.ph==="M"&&e.name==="thread_name")threads.set(key(e),String(e.args?.name||""));
      if(e?.ph==="M"&&e.name==="process_name")processes.set(e.pid,String(e.args?.name||""));
    }
    const events=[],lanes=new Map();let excluded=0;
    for(const [i,e] of raw.entries()){
      if(!e||typeof e!=="object")throw new Error(`事件${i}不是对象`);
      const stage=stageOf(e.name),a=e.args||{};
      if(e.ph!=="X"){
        if(stage)throw new Error(`事件${i} ${e.name} 使用 ${e.ph}；目前只支持完整区间 ph=X`);
        continue;
      }
      const thread=threads.get(key(e))||String(e.tid||"");
      const match=thread.match(/group[_\s-]*(\d+)[_\s-]*(AIC|AIV)[_\s-]*([01])?/i);
      const type=String(a.core_type||match?.[2]||"").toUpperCase();
      const sub=nonnegativeInt(a.sub_id)?Number(a.sub_id):match?.[3]!==undefined?Number(match[3]):type==="AIV1"?1:null;
      const group=nonnegativeInt(a.group_id)?Number(a.group_id):match?Number(match[1]):null;
      const isAiv1=type==="AIV1"||(type==="AIV"&&sub===1);
      if(!isAiv1){
        if(stage&&type!=="AIC"&&!(type==="AIV"&&sub===0))throw new Error(`事件${i} ${e.name} 无法确认 AIV1 分组；需要 args.group_id/core_type/sub_id 或 group_N_AIV_1 线程名`);
        if(stage)excluded++;continue;
      }
      if(group===null)throw new Error(`事件${i} 缺少 AIV1 group_id`);
      const rankMatch=String(e.pid??"").match(/^rank[_\s-]*(\d+)$/i)||processes.get(e.pid)?.match(/rank[_\s-]*(\d+)/i);
      let rank=nonnegativeInt(a.rank)?String(Number(a.rank)):rankMatch?String(Number(rankMatch[1])):null;
      if(rank===null){if(e.pid===undefined)throw new Error(`事件${i} 缺少 Rank/pid`);rank=`pid:${e.pid}`;}
      lanes.set(`${rank}/${group}`,{rank,core:group});
      if(!stage)continue; // WAIT、SYNC、初始化、SQ 不混入 active-family 的累计时间。
      if(typeof e.ts!=="number"||typeof e.dur!=="number"||!Number.isFinite(e.ts)||!Number.isFinite(e.dur)||e.dur<0||!Number.isFinite(e.ts+e.dur))throw new Error(`Rank ${rank} AIV1 ${group} ${e.name} 时间无效`);
      events.push({rank,core:group,stage,name:String(e.name),start:e.ts,duration:e.dur,args:a});
    }
    if(!events.length)throw new Error("没有识别到 AIV1 的 Dispatch/Combine X 事件；WAIT/SYNC 不作为通信事件计入");
    const ranks=[...new Set(events.map(e=>e.rank))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
    if(ranks.some(r=>r.startsWith("pid:")))warnings.push("部分进程只有 pid，未确认物理 Rank；按 pid 分开统计，不跨进程拼接。");
    if(excluded)warnings.push(`排除了${excluded}个非 AIV1 通信事件；本页仅统计 AIV1。`);
    warnings.push("时间来自打点区间，可能含内部等待。WAIT/SYNC 单独事件已排除，隐藏或内部等待无法还原。请确认导入的是一次完整运行，时间裁剪/多次运行不能直接配全量 Shape。");
    return {filename,format,events,lanes:[...lanes.values()],ranks,warnings,rawEventCount:raw.length};
  }
  function unionUs(events){
    const sorted=events.filter(e=>e.duration>0).map(e=>[e.start,e.start+e.duration]).sort((a,b)=>a[0]-b[0]);
    let total=0,lo=null,hi=null;
    for(const [a,b] of sorted){if(lo===null){lo=a;hi=b;}else if(a<=hi)hi=Math.max(hi,b);else{total+=hi-lo;lo=a;hi=b;}}
    return total+(lo===null?0:hi-lo);
  }
  function validate(c){
    for(const k of ["bs","hidden","topk","ep","experts","cores","scaleGroup","fmGroup"])
      if(!Number.isSafeInteger(c[k])||c[k]<1)throw new Error(`${k} 必须为正整数`);
    if(c.cores>128)throw new Error("参与 AIV1 数最多128");
    if(c.topk>c.ep*c.experts)throw new Error("TopK 超过总专家数");
    for(const k of ["dispatchBytes","combineBytes"])if(!Number.isFinite(c[k])||c[k]<=0)throw new Error(`${k} 必须为有限正数`);
    if(!["all","remote","clos"].includes(c.scope))throw new Error("未知字节范围");
    if(!["us","ns","ms"].includes(c.unit))throw new Error("未知 JSON 时间单位");
    if(!Number.isSafeInteger(c.bs*c.topk*c.hidden*Math.max(c.dispatchBytes,c.combineBytes)+c.bs*c.topk*Math.ceil(c.hidden/c.scaleGroup)))throw new Error("Shape 字节数超出安全整数范围；请使用整数 Byte 精度和合理 Shape");
  }
  const bandwidth=(bytes,time)=>time>0?bytes/time/1000:null;
  function analyze(trace,options={},rank=trace.ranks[0]){
    const c={...defaults,...options};validate(c);
    if(!trace.ranks.includes(String(rank)))throw new Error("所选 Rank 不存在");
    rank=String(rank);
    if(!rank.startsWith("pid:")&&Number(rank)>=c.ep)throw new Error(`Rank ${rank} 超出 EP=${c.ep}，请核对 Shape`);
    const factor=trace.format==="MEGAMOE_TRACE_TEXT_V1"?1:{us:1,ns:.001,ms:1000}[c.unit];
    const ev=trace.events.filter(e=>e.rank===rank).map(e=>({...e,start:e.start*factor,duration:e.duration*factor}));
    if(ev.some(e=>e.core>=c.cores))throw new Error(`Rank ${rank} 有 group_id 超出参与 AIV1 数 ${c.cores}；请使用真实核数，不要填写导入片段中的核数`);
    const warnings=[...trace.warnings];
    if(trace.format==="MEGAMOE_TRACE_TEXT_V1"&&c.unit!=="us")warnings.push("文本声明微秒，已忽略 JSON 时间单位选项。");
    const rows=c.bs*c.topk;
    let localGroup=c.fmGroup;
    if(c.scope==="clos"){
      if(rank.startsWith("pid:"))throw new Error("Clos 范围需要已确认的数值 Rank，无法由匿名 pid 判断 FullMesh 组");
      localGroup=Math.min(c.fmGroup,c.ep-Math.floor(Number(rank)/c.fmGroup)*c.fmGroup);
    }
    const fraction=c.scope==="all"?1:c.scope==="remote"?(c.ep-1)/c.ep:(c.ep-localGroup)/c.ep;
    const rowBytes={dispatch:c.hidden*c.dispatchBytes+(c.includeScale?Math.ceil(c.hidden/c.scaleGroup):0),combine:c.hidden*c.combineBytes};
    const volume={dispatch:rows*rowBytes.dispatch*fraction,combine:rows*rowBytes.combine*fraction};
    if(c.scope!=="all")warnings.push("远端/Clos 字节按均衡路由比例估算，并非由 Trace 测得；逐核 Peer 分布可能不均。不能直接当作 NIC 实测吞吐。");
    const cores=[],summary={};
    for(let core=0;core<c.cores;core++){
      const result={core};
      for(const stage of ["dispatch","combine"]){
        const ss=ev.filter(e=>e.core===core&&e.stage===stage),sum=ss.reduce((a,e)=>a+e.duration,0),union=unionUs(ss);
        if(sum-union>Math.max(1e-6,sum*1e-8))warnings.push(`AIV1 ${core} ${stage} 区间重叠/重复：累计${sum.toFixed(3)} μs，并集${union.toFixed(3)} μs；未静默去重。`);
        const bytes=volume[stage]/c.cores;
        result[stage]={count:ss.length,timeUs:sum,unionUs:union,bytes:ss.length?bytes:null,
          gbps:ss.length?bandwidth(bytes,sum):null,rankEquivalentGbps:ss.length?bandwidth(volume[stage],sum):null,
          waves:[...new Set(ss.map(e=>e.args.wave_id).filter(v=>v!==undefined&&v!==null))]};
      }
      cores.push(result);
    }
    for(const stage of ["dispatch","combine"]){
      const ss=ev.filter(e=>e.stage===stage),present=cores.filter(k=>k[stage].count>0),positive=present.filter(k=>k[stage].timeUs>0);
      const sum=present.reduce((a,k)=>a+k[stage].timeUs,0),observedBytes=volume[stage]*present.length/c.cores;
      const union=unionUs(ss),start=ss.reduce((a,e)=>Math.min(a,e.start),Infinity),end=ss.reduce((a,e)=>Math.max(a,e.start+e.duration),-Infinity);
      const complete=present.length===c.cores&&positive.length===c.cores;
      if(!complete)warnings.push(`${stage} 仅${present.length}/${c.cores}核有事件（${positive.length}核累计时间>0）；全核口径的 Rank 汇总不外推，缺核不按0耗时处理。`);
      summary[stage]={count:ss.length,present:present.length,complete,rankBytes:volume[stage],observedBytes,sumCoreUs:sum,
        meanCoreUs:present.length?sum/present.length:null,unionUs:union,envelopeUs:ss.length?end-start:0,
        observedSumGbps:positive.length===present.length&&positive.length?positive.reduce((a,k)=>a+k[stage].gbps,0):null,
        pooledCoreGbps:bandwidth(observedBytes,sum),
        rankEquivalentGbps:complete?bandwidth(volume[stage],sum/c.cores):null,
        unionGbps:complete?bandwidth(volume[stage],union):null,
        envelopeGbps:complete?bandwidth(volume[stage],end-start):null};
    }
    warnings.push("每核字节默认按 Rank 总量/参与 AIV1 数均分，是 Shape 推算，不是 Trace 中的逐核实测字节。Tile/行分工不均时需以真实分工校正；事件次数不用于分配字节。");
    return {version:1,filename:trace.filename,format:trace.format,rank,config:c,rows,rowBytes,fraction,volume,cores,summary,warnings:[...new Set(warnings)],events:ev};
  }
  const api={defaults,parse,analyze,unionUs,stageOf};
  if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.TraceAnalysis=api;
})(globalThis);
