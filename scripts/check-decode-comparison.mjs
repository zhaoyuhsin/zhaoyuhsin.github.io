import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),dir=path.join(root,'decode-optimization');
const page=fs.readFileSync(path.join(dir,'index.html'),'utf8'),script=fs.readFileSync(path.join(dir,'decode.js'),'utf8');
const data=JSON.parse(fs.readFileSync(path.join(dir,'results.json'),'utf8'));
assert.deepEqual(JSON.parse(page.match(/<script id="benchmark-data" type="application\/json">([\s\S]*?)<\/script>/)[1]),data);
assert.equal(data.protocol.device_timer,false);assert.equal(data.protocol.accuracy,'not_run_user_requested');
assert.equal(data.protocol.independent_starts,2);assert.deepEqual(data.cohorts.map(c=>c.ep),[8,32]);
let records=0;
for(const cohort of data.cohorts){
  assert.deepEqual(cohort.cases.map(c=>c.bs),[1,4,8,16,32,96,256,512,1024,2048,4096]);
  for(const item of cohort.cases){
    assert.equal(item.path,item.bs<=512?'decode':'prefill');
    for(const methods of Object.values(item.metrics))for(const method of Object.values(methods))for(const value of Object.values(method)){
      assert.equal(value.starts.length,2);assert.ok(value.starts.every(v=>v>0));
      assert.equal(value.mean,(value.starts[0]+value.starts[1])/2);
      assert.deepEqual(value.range,[Math.min(...value.starts),Math.max(...value.starts)]);
      assert.equal(value.run_spread_pct,100*Math.abs(value.starts[0]-value.starts[1])/value.mean);records++;
    }
  }
}
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{textContent:'',innerHTML:'',handlers:{},attrs:{},children:[],
  addEventListener(event,fn){this.handlers[event]=fn;},setAttribute(k,v){this.attrs[k]=v;},append(p){this.children.push(p);}});return nodes.get(id);}
node('benchmark-data').textContent=JSON.stringify(data);
const buttons=[8,32].map(ep=>Object.assign(node('ep'+ep),{dataset:{ep:String(ep)}}));
const document={getElementById:node,createElement:()=>({}),querySelectorAll:()=>buttons,
  querySelector:selector=>node(selector)};
const context=vm.createContext({document,console});vm.runInContext(script,context);
let combinations=0;
for(const ep of [8,32])for(const scope of ['decode','prefill','all'])for(const metric of ['npu_event_e2e','device_first_last_span'])for(const stat of ['rank_min_us','rank_max_us']){
  buttons.find(b=>Number(b.dataset.ep)===ep).handlers.click();
  for(const [id,value] of [['scope',scope],['metric',metric],['rank-stat',stat]])node(id).handlers.change({target:{value}});
  const cases=data.cohorts.find(c=>c.ep===ep).cases.filter(c=>scope==='all'||c.path===scope);
  const shown=node('#performance-table tbody').innerHTML;
  assert.equal((shown.match(/<tr>/g)||[]).length,cases.length);
  for(const item of cases){
    const m=item.metrics[metric].main_latest[stat],c=item.metrics[metric].cascade[stat];
    const reduction=100*(1-m.mean/c.mean);
    assert.ok(shown.includes((reduction>=0?'+':'−')+Math.abs(reduction).toFixed(2)+'%'));
    assert.ok(shown.includes((c.mean/m.mean).toFixed(2)+'×'));
  }
  for(const id of ['latency-chart','gain-chart'])assert.ok(!/NaN|Infinity|undefined/.test(node(id).innerHTML));
  assert.equal(buttons.find(b=>Number(b.dataset.ep)===ep).attrs['aria-pressed'],'true');combinations++;
}
const metadata=JSON.parse(fs.readFileSync(path.join(dir,'publication.json'),'utf8'));
for(const [name,hash] of Object.entries(metadata.files))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,name))).digest('hex'),hash);
assert.equal(fs.readFileSync(path.join(dir,'results.csv'),'utf8').trim().split('\n').length,89);
for(const name of fs.readdirSync(dir)){
  const content=fs.readFileSync(path.join(dir,name),'utf8');
  assert.ok(!/file:\/\/\/|\b[A-Z]:[\\/]|\/(?:home|root|opt|workspace)\/|\b(?:\d{1,3}\.){3}\d{1,3}\b/i.test(content),'private path/IP '+name);
  assert.ok(!/-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+/.test(content),'credential '+name);
}
for(const match of page.matchAll(/(?:href|src)="([^"<>]+)"/g)){
  const [ref,anchor]=match[1].split('#');if(/^[a-z]+:/i.test(ref))continue;
  let target=ref?path.resolve(dir,ref):path.join(dir,'index.html');
  assert.ok(target===root||target.startsWith(root+path.sep));
  if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  assert.ok(fs.existsSync(target),match[1]);
  if(anchor&&target.endsWith('.html'))assert.ok(fs.readFileSync(target,'utf8').includes(`id="${anchor}"`),match[1]);
}
assert.ok(fs.readFileSync(path.join(root,'index.html'),'utf8').includes('href="decode-optimization/"'));
assert.ok(fs.readFileSync(path.join(root,'megamoe-performance/index.html'),'utf8').includes('href="../decode-optimization/"'));
console.log(`PASS: ${records} exact timing records, ${combinations} interactive filter combinations, formulas/ranges, CSV/embedded JSON, links, SHA and public privacy checks.`);
