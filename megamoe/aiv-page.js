'use strict';
const get = id => document.getElementById(id);
const model = globalThis.AivModel;
const format = x => Number(x).toLocaleString('zh-CN', {maximumFractionDigits:3});
const time = x => format(x)+' μs';
let input;
let result;
let highlighted = 1;

function addField(parent, key, caption, step='1', min='1') {
  const label = document.createElement('label');
  label.textContent = caption;
  const node = document.createElement('input');
  node.name = key;
  node.type = 'number';
  node.step = step;
  node.min = min;
  node.required = true;
  node.value = input[key];
  label.append(node);
  parent.append(label);
}

function reset() {
  input = model.preset(get('preset').value);
  const dq = input.task==='dequant';
  const shape = get('shape');
  const hw = get('hardware');
  shape.replaceChildren();
  hw.replaceChildren();
  addField(shape,'outer',dq?'K':input.task==='swiglu'?'Tile 行数':'本 rank token 数');
  addField(shape,'width',dq?'物理 N':'列宽');
  if (input.task!=='swiglu') addField(shape,'cores',dq?'并行生产者':'可用 AIV 数');
  if (input.task==='unpermute') addField(shape,'topk','TopK');
  if (dq) addField(shape,'block','块大小 / 权重元素');
  addField(shape,'inputSlots',input.task==='unpermute'?'原位槽数':'输入槽数');
  if (input.task!=='unpermute') addField(shape,'outputSlots','输出槽数');
  addField(shape,'overhead','附加开销 / μs','any','0');
  addField(hw,'ghz','AIV / GHz','any','0.000001');
  addField(hw,'read','GM→UB / GB/s','any','0.000001');
  addField(hw,'write',dq?'UB→L1 / GB/s':'UB→GM / GB/s','any','0.000001');
  addField(hw,'readCap','GM 聚合读 / GB/s','any','0');
  if (!dq) addField(hw,'writeCap','GM 聚合写 / GB/s','any','0');
  addField(hw,'ubRead','Vector UB 读 / B每周期','any','0.000001');
  addField(hw,'ubWrite','Vector UB 写 / B每周期','any','0.000001');
  addField(hw,'ubMixed','UB 混合 / B每周期','any','0.000001');
  if (dq) {
    addField(hw,'bit','展开有效 / GOPS','any','0.000001');
  } else {
    addField(hw,'simpleCpi','简单运算 CPI','any','0.000001');
    addField(hw,'castCpi','Cast / 重排 CPI','any','0.000001');
    if (input.task!=='unpermute') addField(hw,'reduceCpi','归约 CPI','any','0.000001');
    if (input.task==='swiglu') {
      addField(hw,'expCpi','Exp CPI','any','0.000001');
      addField(hw,'divCpi','Div CPI','any','0.000001');
    }
  }
  highlighted=1;
  update();
}

function putRows(id, rows) {
  get(id).replaceChildren(...rows.map(values => {
    const row = document.createElement('tr');
    values.forEach(value => { row.insertCell().textContent=value; });
    return row;
  }));
}

function draw() {
  const panel = get('pipeline');
  const first = result.events.filter(e=>e.block<=80);
  const end = first.at(-1).endUs;
  panel.replaceChildren();
  const select = block => {
    highlighted=block;
    panel.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.block)===block)));
    const phases = first.filter(e=>e.block===block).map(e=>e.pipe+' '+format(e.startUs)+'–'+time(e.endUs));
    get('selected').textContent='块 '+block+' · '+phases.join(' · ')+(result.blocks>80?' · 图示前 80 块':'');
  };
  for (const {pipe} of result.services) {
    const lane = document.createElement('div');
    lane.className='lane';
    const label = document.createElement('span');
    label.textContent=pipe;
    const track=document.createElement('div');
    track.className='track';
    for (const e of first.filter(event=>event.pipe===pipe)) {
      const block=document.createElement('button');
      block.type='button';
      block.className='block '+pipe;
      block.style.left=100*e.startUs/end+'%';
      block.style.width=100*(e.endUs-e.startUs)/end+'%';
      block.dataset.block=e.block;
      block.textContent=result.blocks<=32?e.block:'';
      block.title=pipe+' · 块 '+e.block+' · '+format(e.startUs)+'–'+time(e.endUs);
      block.setAttribute('aria-label',block.title);
      block.addEventListener('click',()=>select(e.block));
      track.append(block);
    }
    lane.append(label,track);
    panel.append(lane);
  }
  const ticks=document.createElement('div');
  ticks.className='ticks';
  for (let i=0;i<=4;i++) {
    const tick=document.createElement('span');
    tick.textContent=time(end*i/4);
    ticks.append(tick);
  }
  panel.append(ticks);
  select(Math.min(highlighted,result.blocks,80));
}

function update() {
  try {
    for (const field of get('inputs').querySelectorAll('input')) input[field.name]=field.valueAsNumber;
    if (input.task==='unpermute') input.outputSlots=input.inputSlots;
    result=model.simulate(input);
    get('results').hidden=false;
    get('export').disabled=false;
    get('error').hidden=true;
    get('floor').textContent=time(result.serviceFloorUs);
    get('bound').textContent=result.boundPipe+' 服务时间最长';
    get('finish').textContent=time(result.taskUs);
    get('extra').textContent=result.blocks+' 块 · '+result.active+' 个 AIV 有效';
    const hasRef=result.historicalTaskUs!==null;
    get('third-label').textContent=hasRef?'历史独立微测':'全部串行';
    get('third').textContent=time(hasRef?result.historicalTaskUs:result.serialUs);
    get('third-detail').textContent=hasRef?'32 组并行时单任务中位数':'各段服务时间之和';
    get('scope').textContent=result.scope;
    const writePath=input.task==='dequant'?'UB→L1':'UB→GM';
    putRows('services',[
      ['MTE2 · GM→UB',format(result.readBytes)+' B · '+format(result.effectiveReadGBps)+' GB/s',time(result.services[0].us)],
      ['Vector',result.work.map(([name,value])=>name+' '+format(value)).join(' · '),time(result.services[1].us)],
      ['MTE3 · '+writePath,format(result.writeBytes)+' B · '+format(result.effectiveWriteGBps)+' GB/s',time(result.services[2].us)],
    ]);
    putRows('vector',result.vectorParts.map(p=>[p.name,time(p.issueUs),time(p.ubUs),time(p.serviceUs)]));
    draw();
  } catch (error) {
    get('results').hidden=true;
    get('export').disabled=true;
    get('error').textContent=error.message;
    get('error').hidden=false;
  }
}

get('preset').addEventListener('change',reset);
get('reset').addEventListener('click',reset);
get('inputs').addEventListener('submit',event=>event.preventDefault());
get('inputs').addEventListener('input',event=>{if(event.target.name) update();});
get('export').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify({input,result},null,2)],{type:'application/json'}));
  const link=document.createElement('a');
  link.href=url;
  link.download='megamoe-aiv-model.json';
  link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
reset();
