import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'megamoe-performance');
const html=fs.readFileSync(path.join(out,'index.html'),'utf8');
const data=JSON.parse(fs.readFileSync(path.join(out,'results.json'),'utf8'));
const embedded=JSON.parse(html.match(/<script[^>]*id="benchmark-data"[^>]*>([\s\S]*?)<\/script>/)[1]);
assert.deepEqual(embedded,data);
const docs=JSON.parse(html.match(/<script[^>]*id="project-documents"[^>]*>([\s\S]*?)<\/script>/)[1]);
assert.deepEqual(docs.map(d=>d.id),['project-status','tuning-skill']);
for(const d of docs){
  assert.equal(fs.readFileSync(path.join(out,d.filename),'utf8'),d.markdown);
  assert.equal(crypto.createHash('sha256').update(d.markdown).digest('hex'),d.sha256);
}
assert.ok(docs.every(d=>d.markdown.length>1000),'Public documents must retain their substantive content');
let scripts=0;
for(const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)){
  if(!/application\/json/.test(m[1])){new vm.Script(m[2]);scripts++;}
}
let timingValues=0;
if(process.argv[2]){
  const old=JSON.parse(fs.readFileSync(path.join(path.resolve(process.argv[2]),'results.json'),'utf8'));
  function compare(a,b,p=[]){
    if(typeof a==='number'){assert.equal(b,a,'Numeric measurement changed: '+p.join('.'));timingValues++;}
    else if(Array.isArray(a)){assert.equal(b.length,a.length);a.forEach((v,i)=>compare(v,b[i],p.concat(i)));}
    else if(a&&typeof a==='object')for(const[k,v]of Object.entries(a)){
      if(p.join('.')==='environment'&&k==='files')continue;
      compare(v,b[k],p.concat(k));
    }
  }
  compare(old,data);
}
assert.equal(data.rows.length,7);
assert.equal(data.stage1.ds_converge.latest_original.cases.length,7);
assert.ok(html.includes('id="private-artifacts"'));
assert.ok(html.includes('本地归档（未公开）'));
for(const f of fs.readdirSync(out)){
  assert.ok(['index.html','results.json','PROJECT_STATUS.md','TUNING_SKILL.md','publication.json'].includes(f),'Unexpected published artifact: '+f);
  const text=fs.readFileSync(path.join(out,f),'utf8');
  assert.ok(!/file:\/\/\/|[A-Z]:[\\/]|\/(?:home|root|opt|workspace)\/|\b(?:\d{1,3}\.){3}\d{1,3}\b/i.test(text),'Private path/address: '+f);
  assert.ok(!/-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+/.test(text),'Possible credential: '+f);
}
assert.ok(fs.readFileSync(path.join(root,'index.html'),'utf8').includes('megamoe-performance/index.html'));
console.log(JSON.stringify({passed:true,numeric_values_unchanged:timingValues,views:3,scripts,raw_artifacts_published:false}));
