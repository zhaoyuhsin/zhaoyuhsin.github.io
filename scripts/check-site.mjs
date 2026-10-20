import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),docs=root;
const walk=dir=>fs.readdirSync(dir,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?walk(path.join(dir,entry.name)):[path.join(dir,entry.name)]);
const siteFiles=[path.join(docs,'index.html'),path.join(docs,'404.html'),...walk(path.join(docs,'assets')),...walk(path.join(docs,'megamoe'))];
let localLinks=0,scripts=0;
for(const file of siteFiles){
  const content=fs.readFileSync(file,'utf8');
  if(file.endsWith('.js')){new vm.Script(content,{filename:file});scripts++;}
  if(!file.endsWith('.html'))continue;
  for(const match of content.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi))if(!/\bsrc=/.test(match[1])){new vm.Script(match[2],{filename:file});scripts++;}
  const markup=content.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi,'');
  for(const match of markup.matchAll(/(?:src|href)="([^"<>]+)"/g)){
    if(/^[a-z]+:/i.test(match[1]))continue;
    const [reference,anchor]=match[1].split('#');
    let target=reference.startsWith('/')?path.join(docs,reference):reference?path.resolve(path.dirname(file),reference):file;
    if(!target.startsWith(docs+path.sep)&&target!==docs)throw Error('链接离开网站目录：'+match[1]);
    if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');
    assert.ok(fs.existsSync(target),`${path.relative(docs,file)} missing ${match[1]}`);
    if(anchor&&target.endsWith('.html'))assert.ok(fs.readFileSync(target,'utf8').includes(`id="${decodeURIComponent(anchor)}"`),`missing #${anchor} in ${target}`);
    localLinks++;
  }
  assert.ok(!/file:\/\/\/|(?:C|D):[\\/]/i.test(content),'不应发布本地文件路径：'+file);
}
assert.ok(fs.existsSync(path.join(docs,'.nojekyll')));
assert.ok(!siteFiles.some(file=>/before-|test-|original\.html|rank_\d.*\.json/i.test(path.basename(file))),'备份或真实流水不应出现在发布目录');
// 首页互动使用同一引擎；验证按钮和拓扑变更确实改变比例及带宽阈值。
const sandbox={};vm.runInNewContext(fs.readFileSync(path.join(docs,'assets','ratio-model.js'),'utf8'),sandbox);
const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',style:{},dataset:{},handlers:{},attrs:{},addEventListener(event,handler){this.handlers[event]=handler;},setAttribute(k,v){this.attrs[k]=v;}});return nodes.get(id);};
const buttons=[8,4].map(bits=>{const button=node('bits'+bits);button.dataset.bits=String(bits);return button;});
node('home-topology').value='140';sandbox.document={getElementById:node,querySelectorAll:()=>buttons};
vm.runInNewContext(fs.readFileSync(path.join(docs,'assets','home.js'),'utf8'),sandbox);
assert.equal(node('home-ratio').textContent,'1.347');assert.equal(node('home-target').textContent,'103.9 GB/s');
buttons[1].handlers.click();assert.equal(node('home-ratio').textContent,'0.673');assert.equal(node('home-target').textContent,'207.9 GB/s');
node('home-topology').value='230';node('home-topology').handlers.change();assert.ok(Number(node('home-ratio').textContent)>1);
assert.equal(buttons[1].attrs['aria-pressed'],'true');
console.log(`PASS: ${localLinks} local links/anchors, ${scripts} scripts, public asset allowlist, no file:// paths, homepage precision/topology interactions.`);
