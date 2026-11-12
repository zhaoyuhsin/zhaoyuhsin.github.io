import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const folder='megamoe-performance/';
const data=JSON.parse(read(folder+'shared-expert-comparison.json'));
const page=read(folder+'shared-expert-comparison.html');
assert.equal(data.accuracy,'not_run_user_requested');
assert.equal(data.raw_artifacts_published,false);
assert.equal(data.current_shared.main_promoted,false);
assert.equal(data.current_shared.timer,'all measured invocations, all ranks');
assert.equal(data.historical_no_shared.timer,false);
for(const [kind,expected] of [['current_shared',11],['historical_no_shared',7]]){
  assert.deepEqual(data[kind].results.map(r=>r.ep),[8,32]);
  for(const line of data[kind].results){
    assert.equal(line.rows.length,expected);
    for(const r of line.rows){
      assert.equal(r.slowdown_pct,100*(r.ours.rank_min_us/r.layout_ds.rank_min_us-1));
      for(const m of [r.ours,r.layout_ds]){
        assert.ok(m.rank_min_us>0&&m.rank_max_us>=m.rank_min_us);
        assert.ok(m.rank_min_run_range_us[0]<=m.rank_min_us&&m.rank_min_us<=m.rank_min_run_range_us[1]);
      }
    }
  }
}
// HTML必须直接对应JSON；所有显示值仅在展示时四舍五入，不改保存的测量值。
for(const [kind,id] of [['current_shared','shared-table'],['historical_no_shared','no-shared-table']]){
  const body=page.match(new RegExp(`<table id="${id}">[\\s\\S]*?<tbody>([\\s\\S]*?)</tbody>`))[1];
  const rows=[...body.matchAll(/<tr><th scope="row">(\d+)<\/th>([\s\S]*?)<\/tr>/g)];
  assert.equal(rows.length,data[kind].results[0].rows.length);
  for(const row of rows){
    const bs=Number(row[1]);
    const shown=[...row[2].matchAll(/<td(?: class="[^"]+")?>([^<]+)<\/td>/g)].map(m=>m[1]);
    const expected=data[kind].results.flatMap(line=>{
      const r=line.rows.find(v=>v.bs===bs);
      return [r.ours.rank_min_us.toFixed(2),r.layout_ds.rank_min_us.toFixed(2),
        (r.slowdown_pct>=0?'+':'−')+Math.abs(r.slowdown_pct).toFixed(2)+'%'];
    });
    assert.deepEqual(shown,expected);
  }
}
for(const text of [page,JSON.stringify(data)]){
  assert.ok(!/file:\/\/\/|[A-Z]:[\\/]|\/(?:home|root|opt|workspace)\/|\b(?:\d{1,3}\.){3}\d{1,3}\b/i.test(text));
  assert.ok(!/-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+/.test(text));
}
assert.ok(page.includes('不能相减')&&page.includes('当前版本的无共享专家 vs 布局 DS 对照缺测'));
assert.ok(page.includes('未自动合入 main'));
for(const match of page.matchAll(/(?:src|href)="([^"<>]+)"/g)){
  const [ref,anchor]=match[1].split('#');
  if(/^[a-z]+:/i.test(ref))continue;
  let target=path.resolve(root,folder,ref||'shared-expert-comparison.html');
  if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  assert.ok(target.startsWith(root+path.sep));assert.ok(fs.existsSync(target),match[1]);
  if(anchor)assert.ok(fs.readFileSync(target,'utf8').includes(`id="${anchor}"`),match[1]);
}
// 增量页面不能改变原历史数据、文档或其他实验面板。
const old=execFileSync('git',['show','HEAD:megamoe-performance/index.html'],{cwd:root,encoding:'utf8',maxBuffer:16*1024*1024});
const current=read(folder+'index.html');
for(const id of ['benchmark-data','project-documents']){
  const extract=t=>JSON.parse(t.match(new RegExp(`<script[^>]*id="${id}"[^>]*>([\\s\\S]*?)</script>`))[1]);
  assert.deepEqual(extract(current),extract(old));
}
if(process.argv[2]){
  const source=process.argv[2];
  const audit=JSON.parse(fs.readFileSync(path.join(source,'performance-compare-20261111/summary.json'),'utf8'));
  for(const line of data.current_shared.results)for(const row of line.rows)for(const [key,method] of [['ours','main_latest'],['layout_ds','ds_layout']]){
    const original=audit.lines.find(l=>l.ep===line.ep&&l.shared_experts_per_rank===1).aggregates.find(r=>r.bs===row.bs&&r.method===method);
    assert.equal(row[key].rank_min_us,original.rank_min_us);
    assert.equal(row[key].rank_max_us,original.rank_max_us);
    assert.deepEqual(row[key].rank_min_run_range_us,original.rank_min_run_range_us);
  }
  for(const line of data.historical_no_shared.results){
    const window=line.ep===8?'ep8-r01-8-a':'ep32-r05-1234-c';
    const historical=JSON.parse(fs.readFileSync(path.join(source,'wave-cv-vs-ds-1024',window,'summary.json'),'utf8'));
    for(const row of line.rows)for(const [key,method] of [['ours','combined'],['layout_ds','control']]){
      const original=historical.cases.find(c=>c.bs===row.bs).methods[method];
      assert.equal(row[key].rank_min_us,original.mean_us);
      assert.equal(row[key].rank_max_us,original.rank_max_mean_us);
      assert.equal(original.starts,2);
    }
  }
}
assert.ok(read('index.html').includes('megamoe-performance/shared-expert-comparison.html'));
assert.ok(current.includes('id="shared-comparison-1112"'));
console.log(JSON.stringify({passed:true,current_shared_rows:22,historical_no_shared_rows:14,displayed_values:108,original_dashboard_data_unchanged:true,private_data_published:false}));
