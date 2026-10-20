(function () {
  'use strict';
  const api=globalThis.MegaMoeMemory, $=id=>document.getElementById(id);
  if(!api){$('error').hidden=false;$('error').textContent='缺少 memory-model.js，请保留整个 offline_tool 目录。';$('results').hidden=true;return;}
  const groups={shape:[['bs','bs / Rank'],['H','H'],['N','N（gate+up融合）'],['topk','TopK'],['E','E / EP Rank'],['experts','每卡专家数'],['cores','核组数'],['peak','FP8 Rank TFLOPS']],
    hardware:[['aicRead','单 AIC MTE2 GB/s'],['hbm','Rank HBM读 GB/s'],['miss','HBM读取 miss (%)'],['aiv0Read','单 AIV0 MTE2 GB/s'],['aiv1Read','单 AIV1 MTE2 GB/s'],['aivWrite','单 AIV1 MTE3 GB/s'],['ubToL1','单 AIV0 UB→L1 GB/s'],['mte1','单 AIC MTE1 GB/s'],['fix','单 AIC FIX写回 GB/s'],['ub','单 AIV UB合计读写 GB/s'],['fm','FullMesh每Peer GB/s'],['clos','Clos Rank GB/s'],['netUtil','网络利用率 (%)']],
    ringInputs:[['ringRead','单核本地读有效 GB/s'],['ringWrite','单核发送有效 GB/s'],['ringLatency','发送后完成延迟 μs'],['ringSlots','行环槽数（1～6）'],['ringRows','就绪行数（≤256）']]};
  const fields=Object.values(groups).flat(),key='megamoe.memory.decode.v1';let current=null,restoreWarning='';
  const f=(x,d=2)=>Number.isFinite(x)?x.toLocaleString('zh-CN',{maximumFractionDigits:d}):'待校准';
  const mb=x=>f(x/1048576,3)+' MiB',kb=x=>f(x/1024,2)+' KiB';
  const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const names={aicRead:'MTE2 · GM→L1',aivRead:'MTE2 · GM→UB',aivWrite:'MTE3 · UB→GM',ubToL1:'MTE3 · UB→L1',mte1:'MTE1 · L1→L0',fix:'FIX · L0C→GM',ub:'UB 读+写（下界）'};
  for(const [group,items] of Object.entries(groups)) $(group).innerHTML=items.map(([id,label])=>`<label for="${id}">${label}<input id="${id}" type="number" step="any" placeholder="待校准"></label>`).join('');
  function fill(p){for(const [id] of fields)$(id).value=p[id]??'';}
  function read(){return Object.fromEntries(fields.map(([id])=>[id,$(id).value===''?null:Number($(id).value)]));}
  function band(label,segments){return `<p>${label}</p><div class="track" aria-label="${esc(label)}">${segments.map(([start,end,text,cls])=>`<span class="seg ${cls||''}" style="left:${start/256*100}%;width:${(end-start)/256*100}%" title="${esc(text)}：${f(start)}–${f(end)} KiB">${esc(text)}</span>`).join('')}</div>`;}
  function render(m){
    const {p,a,ub}=m;
    $('metrics').innerHTML=[['每专家 Token / Tile',`${a.M} / ${a.n1} + ${a.n2}`,'GMM1 N128（物理256） / GMM2 N256'],['全部逻辑 GM 读取',mb(m.read),'含已列输入/输出阶段；控制流量未计'],['已知资源下界',f(m.knownFloor)+' μs',m.incomplete?'部分带宽待校准，不能判最终瓶颈':'仍不含完整依赖、Vector计算和控制开销']].map(([l,v,s])=>`<div class="metric">${l}<strong>${v}</strong><small>${s}</small></div>`).join('');
    $('summary').textContent=`bs=${p.bs}、EP${p.E}：共 ${p.E*p.experts} 专家，本 Rank ${p.bs*p.topk} 路由行。GMM1 ${m.tile[0].count} Tile、GMM2 ${m.tile[1].count} Tile。权重搬运不随 M 线性增长：只要同一个 M-group 仍装得下，增加 token 可以摊薄每token的权重成本。`;
    $('ledger').innerHTML=m.rows.map(r=>`<tr><td>${r.stage}<br><b>${r.role}</b></td><td>${names[r.engine]}</td><td class="num">${mb(r.bytes)}</td><td class="num">${kb(r.average)} / ${kb(r.maxCore)}</td><td class="num" title="最忙核 ${r.maxCore} B ÷ (${r.bw??'待校准'} GB/s ×1000)">${f(r.floorUs,3)}</td><td><details><summary>${esc(r.formula)}</summary><div>${f(r.bytes,0)} B / Rank；最忙核 ${f(r.maxCore,0)} B ÷ (${r.bw??'待校准'}×1000) = ${f(r.floorUs,3)} μs。</div></details><small>${esc(r.note)}</small></td></tr>`).join('');
    $('tiles').innerHTML='<div class="scroll"><table><tr><th>Tile</th><th>M×物理N×K</th><th>AIV0 W4读 / W8写</th><th>AIC A+Scale读</th><th>BF16写回</th><th>纯Cube下界</th></tr>'+m.tile.map(t=>`<tr><td>${t.stage}</td><td>${a.M}×256×${t.stage==='GMM1'?p.H:a.I}</td><td>${kb(t.packed)} / ${kb(t.expanded)}</td><td>${kb(t.activation+t.scales)}</td><td>${kb(t.output)}</td><td title="2×M×N×K ÷ (P_Cube/核数×10⁶)">${f(t.cubeUs,3)} μs</td></tr>`).join('')+'</table></div>';
    $('resources').innerHTML='<table><tr><th>常驻角色 / 共享引擎</th><th>跨阶段最忙核合计</th><th>服务时间下界</th></tr>'+m.resource.map(r=>`<tr><td>${r.role} / ${names[r.engine]}</td><td>${mb(r.maxCore)}</td><td>${f(r.floorUs,3)} μs</td></tr>`).join('')+'</table>'+
      '<details><summary>逐核 GMM 数量 / 供数载荷（不含等待）</summary><table><tr><th>核组</th><th>GMM1 / GMM2 Tile</th><th>AIV0 W4 GM读</th><th>AIC GM读</th><th>AIV1 SwiGLU输入读</th></tr>'+a.cores.map(c=>`<tr><td>${c.id}</td><td>${c.g1} / ${c.g2}</td><td>${mb(c.g1*m.tile[0].packed+c.g2*m.tile[1].packed)}</td><td>${mb(c.g1*(m.tile[0].activation+m.tile[0].scales)+c.g2*(m.tile[1].activation+m.tile[1].scales))}</td><td>${kb(c.g1*a.M*128*4)}</td></tr>`).join('')+'</table></details>';
    $('hbm').innerHTML=`T<sub>HBM读</sub> ≥ ${f(m.read,0)} B × ${p.miss}% / (${p.hbm}×1000) = ${f(m.hbmRead,3)} μs<br>已列 GM 写入：${mb(m.write)}（HBM写带宽待校准）<br>纯 Cube Rank 下界：ΣFLOPs / (${p.peak}×10⁶) = ${f(m.cube,3)} μs<br>网络服务：Dispatch ${f(m.dispatchNetwork,3)} μs；Combine ${f(m.combineNetwork,3)} μs<br>T<sub>network</sub> ≥ max(每Peer字节 / W<sub>FM</sub>, Clos字节合计 / W<sub>Clos</sub>) / 网络利用率<br><b>独立资源取最大值给出必要下界，不把重叠区间或读写字节直接相加成时间。</b>`;
    $('buffers').innerHTML='<p class="muted">横轴为每个 AIV 自己的 UB 字节地址：0～256 KiB。各行是不同时刻的布局，不是并发相加。</p>'+band('AIV0 · 权重展开（GMM1/GMM2复用）',[[0,64,'W4 四槽'],[64,192,'W8 四槽','v']])+`<p>预留 ${kb(ub.aiv0Reserved)}；源码该 Prologue 自检上限248 KiB。两半 K 顺序展开，L1 双缓冲没有释放时 AIV0 停等。</p>`+
      band(`AIV1 · Dispatch：${ub.dispatchSlots}槽，布局合计 ${kb(ub.dispatchBytes)}`,[[0,ub.dispatchBytes/1024,'prefix / route / 行环 / metadata']])+
      band('AIV1 · SwiGLU输入视图',[[0,128,'gate：预留128KiB','v'],[128,256,'up：预留128KiB','v']])+
      `<p>当前完整 Tile 的有效输入仅 ${kb(ub.swPayload)}；输入最高访问地址约 ${kb(ub.swAddressEnd)}。物理行stride为256列，实际只用128列。激活/量化临时视图复用低地址，视图最大端点 ${kb(ub.swScratchEnd)}；因此不能把两路输入和所有输出数组容量相加。</p>`+
      band('AIV1 · Token Combine',[[64,64+ub.combineRing/1024,'6槽BF16行环','c'],[184,200,'ready区域','v'],[200,248,'metadata视图','c']])+
      `<p>行环 ${kb(ub.combineRing)}；metadata预留48 KiB，当前一个均分slice每核约${f(a.M/p.cores)}行、实际meta约 ${f(ub.combineMetaActual)} B。Dispatch/Combine 返回前排空行环，才可由 SwiGLU 重用 UB。</p>`;
    const b=m.boundary;$('boundary').innerHTML=`<div class="formula">输入量化：读 bs×H×2 = ${mb(b.inputRead)}；写 bs×Dispatch记录 = ${mb(b.inputWrite)}<br>Unpermute：读 bs×TopK×(2H+4B权重) ≳ ${mb(b.unpermuteRead)}；写 bs×H×2 = ${mb(b.unpermuteWrite)}</div>`;
    if(!m.ring){$('ringResult').innerHTML='<p class="notice">请填写单核读/发送带宽，或使用明确标为“非实测”的演示值。未填时不生成虚假的时间条。</p>';return;}
    const r=m.ring,limit=Math.max(r.total,0.001),x=t=>130+t/limit*750;
    const shown=r.events.slice(0,24);
    $('ringResult').innerHTML=`<p>每行 ${kb(p.H*2)}；${p.ringRows} 行，${p.ringSlots}槽：局部完成 ${f(r.total,3)} μs；MTE2因槽未释放停等 ${f(r.blocked,3)} μs。后者只统计此行环，不是全算子气泡。</p><div class="scroll"><svg viewBox="0 0 920 ${50+shown.length*27}" role="img" aria-label="Combine行环的读取、发送和完成延迟"><text x="130" y="16" font-size="12">0 μs · 蓝=读入　青=发送　灰=等发送完成/释放槽</text><text x="880" y="32" text-anchor="end" font-size="12">${f(limit,3)} μs</text>${shown.map((e,i)=>{const y=42+i*27;return `<text x="5" y="${y+14}" font-size="12">行${e.row} / 槽${e.row%p.ringSlots}</text>`+[[e.readStart,e.readEnd,'#3555d5','MTE2读'],[e.writeStart,e.writeEnd,'#118b8e','MTE3发送'],[e.writeEnd,e.release,'#b8c2cf','完成延迟']].map(([s,t,c,l])=>`<rect x="${x(s)}" y="${y}" width="${Math.max(0,(t-s)/limit*750)}" height="18" fill="${c}"><title>${l}：${f(s,3)}–${f(t,3)} μs；${e.row>=p.ringSlots?'须等待行'+(e.row-p.ringSlots)+'释放同槽':'首轮空闲槽'}</title></rect>`).join('');}).join('')}</svg></div>${p.ringRows>24?'<p>图只显示前24行，统计覆盖全部行。</p>':''}`;
  }
  function update(){try{const m=api.model(read());current=m;render(m);$('error').hidden=true;$('results').hidden=false;
    try{localStorage.setItem(key,JSON.stringify({version:1,params:m.p}));$('status').textContent=restoreWarning+'配置已保存到当前浏览器；不会覆盖主流水页配置。';}catch(e){$('status').textContent=restoreWarning+'浏览器存储不可用；可导出JSON保留配置：'+e.message;}
  }catch(e){current=null;$('error').hidden=false;$('error').textContent=e.message;$('results').hidden=true;}}
  fill(api.defaults);
  try{const raw=localStorage.getItem(key);if(raw){const saved=JSON.parse(raw);if(saved.version!==1)throw Error('配置版本不兼容');const p={...api.defaults,...saved.params};api.model(p);fill(p);}}catch(e){restoreWarning='本地配置无法恢复，已使用默认值：'+e.message+'。';}
  for(const [id] of fields)$(id).addEventListener('change',update);
  $('reset').addEventListener('click',()=>{fill(api.defaults);update();});
  $('demo').addEventListener('click',()=>{$('ringRead').value=100;$('ringWrite').value=4.375;$('ringLatency').value=1;$('ringRows').value=18;update();});
  $('export').addEventListener('click',()=>{if(!current){$('error').textContent='请先修正输入，不能导出无效结果。';return;}
    const blob=new Blob([JSON.stringify(current,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='decode-master-mte-ub.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  update();
})();
