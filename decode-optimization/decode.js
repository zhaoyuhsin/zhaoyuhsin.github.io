/* 所有可视化和比例都从同一公开快照派生；切换统计不挑选各次启动的最低值。 */
const benchmark=JSON.parse(document.getElementById('benchmark-data').textContent);
const view={ep:8,scope:'decode',metric:'npu_event_e2e',stat:'rank_min_us'};
const number=v=>v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const signed=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(2)+'%';
const gain=(m,c)=>100*(1-m/c);
const color=v=>Math.abs(v)<1?'near':v>0?'faster':'slower';
const metricName=()=>view.metric==='npu_event_e2e'?'NPU Event 端到端':'设备首尾跨度';
const statName=()=>view.stat==='rank_min_us'?'rank-min':'rank-max';
const cohort=()=>benchmark.cohorts.find(c=>c.ep===view.ep);
const selected=()=>cohort().cases.filter(c=>view.scope==='all'||c.path===view.scope);
const times=item=>{const m=item.metrics[view.metric];return [m.main_latest[view.stat],m.cascade[view.stat]];};
const svgText=(x,y,text,extra='')=>`<text x="${x}" y="${y}" ${extra}>${text}</text>`;

function plot(rows,type){
  const w=550,h=300,left=57,right=22,top=25,bottom=48,pw=w-left-right,ph=h-top-bottom;
  const lo=Math.log2(rows[0].bs),hi=Math.log2(rows.at(-1).bs);
  const x=bs=>left+(Math.log2(bs)-lo)/Math.max(1,hi-lo)*pw;
  const max=type==='latency'?Math.max(...rows.flatMap(r=>times(r).flatMap(v=>v.range)))*1.13:Math.max(10,...rows.map(r=>gain(...times(r).map(v=>v.mean))))*1.15;
  const min=type==='latency'?0:Math.min(-5,...rows.map(r=>gain(...times(r).map(v=>v.mean))))*1.2;
  const y=v=>top+(max-v)/(max-min)*ph;
  let html=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="EP${view.ep} ${type==='latency'?'完整耗时曲线':'耗时收益曲线'}，数据见下方表格"><title>EP${view.ep} ${metricName()} ${statName()}</title>`;
  for(let i=0;i<=4;i++){
    const value=min+(max-min)*i/4,pos=y(value);
    html+=`<line x1="${left}" x2="${w-right}" y1="${pos}" y2="${pos}" stroke="#e4eae2"/>`;
    html+=svgText(left-10,pos+4,type==='latency'?Math.round(value).toLocaleString('en-US'):value.toFixed(0)+'%','text-anchor="end"');
  }
  if(type==='gain')html+=`<line x1="${left}" x2="${w-right}" y1="${y(0)}" y2="${y(0)}" stroke="#8d9e91" stroke-dasharray="4 4"/>`;
  for(const r of rows)html+=svgText(x(r.bs),h-bottom+22,String(r.bs),'text-anchor="middle"');
  html+=svgText(w-right,h-5,'BS / 卡','text-anchor="end"');
  const series=type==='latency'?[{index:0,name:'MegaMoE',stroke:'#24634e'},{index:1,name:'小算子拼接',stroke:'#b37a39'}]:[{name:'耗时收益',stroke:'#24634e'}];
  for(const s of series){
    const values=rows.map(r=>type==='latency'?times(r)[s.index].mean:gain(...times(r).map(v=>v.mean)));
    html+=`<polyline points="${rows.map((r,i)=>`${x(r.bs)},${y(values[i])}`).join(' ')}" fill="none" stroke="${s.stroke}" stroke-width="2.5"/>`;
    rows.forEach((r,i)=>{
      const pos=x(r.bs),val=values[i];
      if(type==='latency'){
        const range=times(r)[s.index].range;
        html+=`<path d="M${pos},${y(range[0])}V${y(range[1])} M${pos-4},${y(range[0])}H${pos+4} M${pos-4},${y(range[1])}H${pos+4}" stroke="${s.stroke}" fill="none"/>`;
      }
      html+=`<circle cx="${pos}" cy="${y(val)}" r="4" fill="${type==='gain'&&val<0?'#a24c36':s.stroke}" stroke="white" stroke-width="1.4"><title>BS${r.bs} · ${s.name}: ${type==='latency'?number(val)+' μs':signed(val)}</title></circle>`;
    });
  }
  return html+'</svg>';
}

function render(){
  const rows=selected(),scope=view.scope==='all'?'全部 BS':view.scope;
  const title=`EP${view.ep} · ${scope} · ${metricName()} · ${statName()}`;
  document.getElementById('selection-summary').textContent=title;
  document.getElementById('table-caption').textContent=title+' · 双轮均值 · 单位 μs';
  for(const b of document.querySelectorAll('[data-ep]'))b.setAttribute('aria-pressed',String(Number(b.dataset.ep)===view.ep));
  const gains=rows.map(r=>gain(...times(r).map(v=>v.mean))),wins=gains.filter(v=>v>0).length;
  const peak=Math.max(...gains),best=rows[gains.indexOf(peak)].bs;
  const spread=Math.max(...rows.map(r=>times(r)[1].run_spread_pct));
  document.getElementById('insights').innerHTML=`<div class="insight"><span>本筛选范围 · MegaMoE 更快</span><strong>${wins} / ${rows.length} 个 BS</strong><small>只计双轮均值，不判断显著性</small></div><div class="insight"><span>最大耗时收益 · BS${best}</span><strong class="${color(peak)}">${signed(peak)}</strong><small>${view.scope==='all'?'decode / prefill 请按路径区分':scope+' 路径，非跨 BS 平均收益'}</small></div><div class="insight"><span>拼接 · 最大两次均值差异</span><strong>${spread.toFixed(2)}%</strong><small>不是单轮抖动或置信区间</small></div>`;
  document.getElementById('latency-chart').innerHTML=plot(rows,'latency');
  document.getElementById('gain-chart').innerHTML=plot(rows,'gain');
  document.querySelector('#performance-table tbody').innerHTML=rows.map(r=>{
    const [m,c]=times(r),benefit=gain(m.mean,c.mean);
    const cell=v=>`<td>${number(v.mean)}<small class="run-values">${number(v.starts[0])} / ${number(v.starts[1])}</small></td>`;
    return `<tr><th scope="row">${r.bs}</th><td><span class="path-tag ${r.path}">${r.path}</span></td>${cell(m)}${cell(c)}<td class="${color(benefit)}">${signed(benefit)}</td><td>${(c.mean/m.mean).toFixed(2)}×</td><td>${m.run_spread_pct.toFixed(2)}%</td><td>${c.run_spread_pct.toFixed(2)}%</td></tr>`;
  }).join('');
}
for(const b of document.querySelectorAll('[data-ep]'))b.addEventListener('click',()=>{view.ep=Number(b.dataset.ep);render();});
for(const [id,key] of [['scope','scope'],['metric','metric'],['rank-stat','stat']])document.getElementById(id).addEventListener('change',event=>{view[key]=event.target.value;render();});
// 来源只发布逻辑窗口ID和不可变审计指纹，绝不嵌入本机路径或节点地址。
for(const c of benchmark.cohorts){const p=document.createElement('p');p.textContent=`EP${c.ep} · ${c.source_id} · audit SHA256 ${c.audit_sha256}`;p.className='hash';document.getElementById('source-hashes').append(p);}
render();
