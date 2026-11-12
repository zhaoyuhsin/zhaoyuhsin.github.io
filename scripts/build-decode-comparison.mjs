import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const site=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const evidence=process.argv[2];
if(!evidence)throw Error('Usage: node scripts/build-decode-comparison.mjs <outputs-directory>');
const bsList=[1,4,8,16,32,96,256,512,1024,2048,4096];
const sources=[[8,'performance-compare-20261111-r05-f'],[32,'performance-compare-20261111-r05-i']];
const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
const sha=raw=>crypto.createHash('sha256').update(raw).digest('hex');
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-8*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const data={schema_version:1,published:'2026-11-12',measured:'2026-11-11',
  title:'MegaMoE vs 小算子拼接 · 无共享专家',
  model:{hidden:7168,intermediate_hidden:3072,num_experts:384,num_topk:6,num_shared_experts:0},
  input:{bs_definition:'per rank',routing:'balanced_rank_first',seed:20260922,routed_weights:'FP4 E2M1 / E8M0 scale',activation:'FP8'},
  protocol:{independent_starts:2,warmup:10,sampled_rounds:20,discard_first:10,
    device_timer:false,profiler:'Level1 / PipeUtilization',accuracy:'not_run_user_requested',
    default_metric:'npu_event_e2e',default_rank_stat:'rank_min_us',
    rank_reduction:'Reduce ranks within each retained round, average 10 rounds, then average two independent starts.',
    latency_reduction:'100 * (cascade - main) / cascade',speedup:'cascade / main',
    run_spread:'100 * abs(start_a - start_b) / mean(start_a, start_b)'},
  versions:{main:'c5d68f0d174be6a0fb81676de560e2193dd11e35 + shared-fp8-1110 isolated candidate; not promoted to main',
    main_kernel_sha256:'92abc16f4f8f70fe29f42ee2771b633b8d8ad6129a733b2978861a99714f120d',
    cascade:'ad369c0ac817f36ad6f2306f1c2e4834b6ad4568 + binding adapter 3f95223',
    cascade_repository:'https://gitcode.com/SimpleBright_Man/mega_test',
    environment:'Ascend950DT / CANN 9.2.0 B080 / torch 2.12.0+cpu / torch-npu 2.12.0.post2'},
  limitations:['Profiler-on eager serial reference; not graph or multistream engine performance.',
    'No device-internal timer trace in this matched S=0 campaign.',
    'Only two independent starts; no statistical significance or long-term stability claim.',
    'Decode path: BS <= 512. Prefill path: BS > 512; do not call prefill gains decode gains.',
    'EP8 uses per-BS main buffer capacity; EP32 reuses one capacity-4096 buffer. Cross-EP difference is not a single-variable EP effect.',
    'Performance only, accuracy intentionally not tested. Do not infer production merge readiness.'],cohorts:[]};

for(const [ep,window] of sources){
  const raw=fs.readFileSync(path.join(evidence,window,`audit-ep${ep}.json`));
  const audit=JSON.parse(raw);
  assert.equal(audit.status,'audited_performance_only');
  assert.equal(audit.accuracy,'not_run_user_requested');
  assert.equal(audit.line_status.shared0.complete,true);
  const cohort={ep,placement:'same rack, matched ordered nodes for candidate and reference',
    source_id:window,audit_sha256:sha(raw),hccl_buffer_mb:6000,
    main_buffer_policy:ep===8?'per_bs_capacity':'session_reuse_capacity4096',cases:[]};
  for(const bs of bsList){
    const result={bs,path:bs<=512?'decode':'prefill',metrics:{}};
    for(const metric of ['npu_event_e2e','device_first_last_span']){
      const values={};
      for(const method of ['main_latest','cascade']){
        const rows=audit.rows.filter(r=>r.shared===0&&r.bs===bs&&r.method===method&&r.metric===metric);
        assert.equal(rows.length,2);assert.deepEqual(rows.map(r=>r.start).sort(),['a','b']);
        assert.ok(rows.every(r=>r.session_complete));
        const agg=audit.aggregates.find(r=>r.shared===0&&r.bs===bs&&r.method===method&&r.metric===metric);
        assert.equal(agg.independent_starts,2);
        values[method]={};
        for(const stat of ['rank_min_us','rank_max_us']){
          const starts=['a','b'].map(s=>rows.find(r=>r.start===s)[stat]);
          assert.ok(starts.every(v=>Number.isFinite(v)&&v>0));
          close(mean(starts),agg[stat]);
          values[method][stat]={mean:mean(starts),starts,range:[Math.min(...starts),Math.max(...starts)],
            run_spread_pct:100*Math.abs(starts[0]-starts[1])/mean(starts)};
        }
      }
      result.metrics[metric]=values;
      const comp=audit.comparisons.find(r=>r.shared===0&&r.bs===bs&&r.metric===metric&&r.reference==='cascade');
      close(100*(1-values.main_latest.rank_min_us.mean/values.cascade.rank_min_us.mean),comp.rank_min_latency_reduction_pct);
    }
    cohort.cases.push(result);
  }
  data.cohorts.push(cohort);
}

const out=path.join(site,'decode-optimization');fs.mkdirSync(out,{recursive:true});
const json=JSON.stringify(data,null,2)+'\n';
fs.writeFileSync(path.join(out,'results.json'),json);
const csv=['ep,bs_per_rank,path,metric,rank_stat,main_us,cascade_us,latency_reduction_pct,speedup,main_start_a_us,main_start_b_us,cascade_start_a_us,cascade_start_b_us,main_run_spread_pct,cascade_run_spread_pct'];
for(const cohort of data.cohorts)for(const item of cohort.cases)for(const [metric,methods] of Object.entries(item.metrics))for(const stat of ['rank_min_us','rank_max_us']){
  const m=methods.main_latest[stat],c=methods.cascade[stat];
  csv.push([cohort.ep,item.bs,item.path,metric,stat,m.mean,c.mean,100*(1-m.mean/c.mean),c.mean/m.mean,...m.starts,...c.starts,m.run_spread_pct,c.run_spread_pct].join(','));
}
fs.writeFileSync(path.join(out,'results.csv'),csv.join('\n')+'\n');
const fallback=data.cohorts[0].cases.filter(r=>r.path==='decode').map(r=>{
  const v=r.metrics.npu_event_e2e,m=v.main_latest.rank_min_us.mean,c=v.cascade.rank_min_us.mean;
  return `<tr><th scope="row">${r.bs}</th><td>decode</td><td>${m.toFixed(2)}</td><td>${c.toFixed(2)}</td><td>${(100*(1-m/c)).toFixed(2)}%</td><td>${(c/m).toFixed(2)}×</td><td>${v.main_latest.rank_min_us.run_spread_pct.toFixed(2)}%</td><td>${v.cascade.rank_min_us.run_spread_pct.toFixed(2)}%</td></tr>`;
}).join('\n');
const template=fs.readFileSync(path.join(site,'scripts/decode-optimization.template.html'),'utf8');
assert.ok(template.includes('<!--RESULTS-->')&&template.includes('<!--FALLBACK-->'));
fs.writeFileSync(path.join(out,'index.html'),template.replace('<!--RESULTS-->',json.replaceAll('<','\\u003c')).replace('<!--FALLBACK-->',fallback));
fs.writeFileSync(path.join(out,'publication.json'),JSON.stringify({published:data.published,sources:data.cohorts.map(c=>({ep:c.ep,id:c.source_id,audit_sha256:c.audit_sha256})),
  files:Object.fromEntries(['index.html','decode.css','decode.js','results.json','results.csv'].map(n=>[n,sha(fs.readFileSync(path.join(out,n)))])),
  public_scope:'Sanitized aggregate performance only; no raw trace, logs, machine addresses, local paths, or credentials.'},null,2)+'\n');
console.log('Built: 22 paired cases, 2 metrics, rank-min/rank-max, two independent starts; sanitized publication.');
