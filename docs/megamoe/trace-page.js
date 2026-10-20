(function(){
  "use strict";
  const A=TraceAnalysis,$=id=>document.getElementById(id),storageKey="megamoe.trace-analysis.settings.v1";
  $("defaultSummary").textContent=`默认：EP${A.defaults.ep}、每卡${A.defaults.experts}专家、BS${A.defaults.bs}、H${A.defaults.hidden}、TopK${A.defaults.topk}、均衡路由、${A.defaults.cores}核组；Combine不量化（${A.defaults.combineBytes} Byte/元素）。`;
  let trace=null,result=null,generation=0,storageError="";
  const fmt=(v,n=3)=>v===null||v===undefined||!Number.isFinite(v)?"—":v.toLocaleString("zh-CN",{maximumFractionDigits:n,minimumFractionDigits:n});
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function read(){return Object.fromEntries(Object.entries(A.defaults).map(([k,v])=>[k,typeof v==="boolean"?$(k).checked:typeof v==="number"?Number($(k).value):$(k).value]));}
  function apply(c){for(const [k,v] of Object.entries(A.defaults)){if(typeof v==="boolean")$(k).checked=c[k]??v;else $(k).value=c[k]??v;}}
  function message(text,error=false){$("status").textContent=text;$("status").classList.toggle("error",error);}
  function invalidate(){result=null;$("results").classList.add("hidden");$("csv").disabled=$("json").disabled=true;}
  function render(r){
    const c=r.config,d=r.summary.dispatch,b=r.summary.combine;
    $("volume").textContent=`Rank ${r.rank} · ${r.rows.toLocaleString()} 路由行 · Dispatch ${fmt(r.volume.dispatch/1048576)} MiB / Combine ${fmt(r.volume.combine/1048576)} MiB · 范围比例 ${fmt(r.fraction*100)}% · 均分 ${c.cores} 个 AIV1`;
    $("metrics").innerHTML=[
      ["Dispatch 累计核时",d.sumCoreUs,"μs",`已导入 ${d.present}/${c.cores} 核；不是墙钟时间`,""],
      ["Dispatch 逐核带宽之和",d.observedSumGbps,"GB/s","Σ 每核分摊字节 / 每核累计时间",""],
      ["Combine 累计核时",b.sumCoreUs,"μs",`已导入 ${b.present}/${c.cores} 核；不是墙钟时间`,"combine"],
      ["Combine 逐核带宽之和",b.observedSumGbps,"GB/s","Σ 每核分摊字节 / 每核累计时间","combine"]
    ].map(([label,value,unit,hint,cls])=>`<div class="metric ${cls}"><div>${label}</div><div class="value">${fmt(value)} <small>${unit}</small></div><div class="hint">${hint}</div></div>`).join("");
    $("coreTable").querySelector("tbody").innerHTML=r.cores.map(k=>`<tr class="${!k.dispatch.count&&!k.combine.count?"missing":""}"><td>${k.core}</td>`+["dispatch","combine"].map(stage=>{
      const s=k[stage],tip=`Vrank=${r.volume[stage]} B；Vᵢ=Vrank/${c.cores}；Tᵢ=${s.timeUs} μs；分摊GB/s=Vᵢ/Tᵢ/1000。${s.count?"按完整一次运行的均分量估算":"未记录此阶段"}`;
      return `<td>${s.count||"缺失"}</td><td>${s.count?fmt(s.timeUs):"—"}</td><td title="${esc(tip)}">${fmt(s.bytes===null?null:s.bytes/1048576)}</td><td title="${esc(tip)}">${fmt(s.gbps)}</td><td title="${esc(`全Rank字节 ${r.volume[stage]} / ${s.timeUs} μs /1000，不是单核实际吞吐`)}">${fmt(s.rankEquivalentGbps)}</td>`;
    }).join("")+"</tr>").join("");
    const lines=[
      ["有事件核数 / 配置核数",s=>`${s.present} / ${c.cores}`],
      ["事件数",s=>String(s.count)],
      ["Shape 全Rank通信量 MiB",s=>fmt(s.rankBytes/1048576)],
      ["已导入核分摊通信量 MiB（估算）",s=>fmt(s.observedBytes/1048576)],
      ["Σ 累计核时 μs",s=>fmt(s.sumCoreUs)],
      ["已导入核平均累计时间 μs",s=>fmt(s.meanCoreUs)],
      ["Σ 已导入核分摊带宽 GB/s",s=>fmt(s.observedSumGbps)],
      ["分摊累计核时带宽 GB/s",s=>fmt(s.pooledCoreGbps)],
      ["全核平均累计时间 Rank折算 GB/s",s=>fmt(s.rankEquivalentGbps)],
      ["同阶段跨核区间并集 μs",s=>fmt(s.unionUs)],
      ["全Rank / 并集时间 GB/s",s=>fmt(s.unionGbps)],
      ["同阶段首尾包络 μs",s=>fmt(s.envelopeUs)],
      ["全Rank / 首尾包络 GB/s",s=>fmt(s.envelopeGbps)]
    ];
    $("summaryTable").querySelector("tbody").innerHTML=lines.map(([name,f])=>`<tr><td>${name}</td><td>${f(d)}</td><td>${f(b)}</td></tr>`).join("");
    $("warnings").innerHTML=r.warnings.map(w=>`<li>${esc(w)}</li>`).join("");
    $("results").classList.remove("hidden");$("csv").disabled=$("json").disabled=false;
  }
  function run(){
    if(!trace){message("请先导入流水文件。");return;}
    try{
      result=A.analyze(trace,read(),$("rank").value);
      try{localStorage.setItem(storageKey,JSON.stringify(result.config));}catch{storageError="设置无法保存到本浏览器；当前分析仍可用。";}
      render(result);
      const incomplete=["dispatch","combine"].filter(k=>!result.summary[k].complete);
      message(`${trace.filename} · ${trace.format} · Rank ${result.rank}\n${result.events.length} 个通信事件；${incomplete.length?"存在缺核/零耗时：全 Rank 带宽不外推。":"全核有记录；仍需确认时间上是完整单次运行。"}${storageError?"\n"+storageError:""}`);
    }catch(e){invalidate();message(e.message,true);}
  }
  function load(text,name){
    invalidate();trace=null;$("rankControls").classList.add("hidden");
    try{trace=A.parse(text,name);$("rank").innerHTML=trace.ranks.map(r=>`<option value="${esc(r)}">${esc(r)}</option>`).join("");$("rankControls").classList.remove("hidden");run();}
    catch(e){message(e.message,true);}
  }
  async function fileLoad(files){
    const version=++generation;invalidate();trace=null;$("rankControls").classList.add("hidden");
    if(files.length!==1){message("请每次只导入一个文件；多Rank可放在同一个Trace JSON中。",true);return;}
    const f=files[0];if(f.size>100*1024*1024){message("文件超过100 MiB，请按单次运行、单Rank导出。",true);return;}
    message(`正在读取 ${f.name}…`);
    try{const text=await f.text();if(version===generation)load(text,f.name);}catch(e){if(version===generation)message(`读取失败：${e.message}`,true);}
  }
  function download(name,type,body){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $("csv").addEventListener("click",()=>{
    if(!result)return;
    const headers=["rank","aiv1","stage","count","sum_us","estimated_bytes","estimated_core_gbps","rank_equivalent_gbps","bs","H","topk","EP","aiv1_count","scope","include_scale"];
    const rows=result.cores.flatMap(k=>["dispatch","combine"].map(stage=>{const s=k[stage],c=result.config;return [result.rank,k.core,stage,s.count,s.timeUs,s.bytes,s.gbps,s.rankEquivalentGbps,c.bs,c.hidden,c.topk,c.ep,c.cores,c.scope,c.includeScale];}));
    const cell=v=>'"'+String(v??"").replace(/^[=+@-]/,"'$&").replace(/"/g,'""')+'"';
    download("megamoe-trace-bandwidth.csv","text/csv;charset=utf-8","\uFEFF"+[headers,...rows].map(row=>row.map(cell).join(",")).join("\r\n"));
  });
  $("json").addEventListener("click",()=>{if(result)download("megamoe-trace-analysis.json","application/json",JSON.stringify({...result,exportedAt:new Date().toISOString(),byteMethod:"Shape full-rank volume / configured AIV1 count; balanced base routing estimate",bandwidthUnit:"decimal GB/s",timeUnit:"us"},null,2));});
  $("file").addEventListener("change",e=>fileLoad(e.target.files));
  const drop=$("drop");for(const name of ["dragover","dragenter"])drop.addEventListener(name,e=>{e.preventDefault();drop.classList.add("over");});
  drop.addEventListener("dragleave",()=>drop.classList.remove("over"));drop.addEventListener("drop",e=>{e.preventDefault();drop.classList.remove("over");fileLoad(e.dataTransfer.files);});
  // 阻止误拖到页面其他位置时浏览器直接导航离开工具。
  window.addEventListener("dragover",e=>e.preventDefault());window.addEventListener("drop",e=>e.preventDefault());
  $("parsePaste").addEventListener("click",()=>{generation++;load($("paste").value,"粘贴流水");});
  $("shapeForm").addEventListener("submit",e=>{e.preventDefault();run();});$("shapeForm").addEventListener("change",()=>{if(trace)run();});$("rank").addEventListener("change",run);
  $("resetDefaults").addEventListener("click",()=>{
    apply(A.defaults);
    try{localStorage.setItem(storageKey,JSON.stringify(A.defaults));storageError="";}catch{storageError="默认配置已恢复，但无法保存到本浏览器。";}
    if(trace)run();else message(storageError||`已恢复默认配置。${$("defaultSummary").textContent}`);
  });
  $("readShape").addEventListener("click",()=>{
    try{const saved=JSON.parse(localStorage.getItem("megamoe.offline.last-run.v1")||"null")?.values;if(!saved)throw new Error("当前浏览器没有仿真页上次运行配置；file:// 与 localhost 的存储可能不互通");
      if(saved.commVariant&&saved.commVariant!=="base")throw new Error("仿真页使用去冗余分支，本页仅支持 Base 字节公式；未导入");
      const c=read();for(const k of ["bs","hidden","topk","ep","cores","fmGroup"])if(saved[k]!==undefined)c[k]=Number(saved[k]);if(saved.expertsPerRank!==undefined)c.experts=Number(saved.expertsPerRank);apply(c);if(trace)run();else message("已读取仿真页 Shape；精度、Scale及字节范围请独立核对。");
    }catch(e){message(e.message,true);}
  });
  try{apply({...A.defaults,...JSON.parse(localStorage.getItem(storageKey)||"{}")});}catch{apply(A.defaults);message("无法恢复本地设置，已使用默认 Shape。");}
})();
