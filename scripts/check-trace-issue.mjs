import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const html=fs.readFileSync(path.join(root,'megamoe/trace-v2.html'),'utf8');
const start=html.indexOf('(function (root)'),marker=html.indexOf('root.TraceAnalysisV2 = api;');
assert.ok(start>=0&&marker>start,'缺少内联分析引擎');
const end=html.indexOf('\n',html.indexOf('})(typeof globalThis',marker));
const sandbox={};vm.runInNewContext(html.slice(start,end),sandbox);
const A=sandbox.TraceAnalysisV2;
const event=(name,ts,dur,kind='AIC',core=0,args={})=>({ph:'X',pid:'rank0',tid:`group_${core}_${kind==='AIC'?'AIC_0':'AIV_1'}`,
  name,ts,dur,args:{rank:0,group_id:core,core_type:kind==='AIC'?'AIC':'AIV',sub_id:kind==='AIC'?0:1,time_origin:'rank',...args}});
const config={ep:32,totalExperts:384,bs:4096,aicCores:2,cores:2};
const analyze=events=>A.analyze(A.parse(JSON.stringify(events),'synthetic.json',{rank0Only:true}),config);
const baseline=[];
for(let core=0;core<2;core++)baseline.push(
  event('AIV_INPUT_SYNC',0,1,'AIV',core),event('wave_0_AIV_DISPATCH_0',1,5,'AIV',core),
  event('wave_0_AIC_GMM1_0',6,20,'AIC',core),event('wave_0_AIV_SWIGLU_0',26,3,'AIV',core),
  event('wave_0_AIC_GMM2_0',29,10,'AIC',core),event('wave_0_AIV_COMBINE_0',39,8,'AIV',core),
  event('AIV_OUTPUT_SYNC',47,1,'AIV',core));
const legacy=analyze(baseline);
assert.equal(legacy.stages.gemm1.count,2);assert.equal(legacy.cube.activeCoreUs,60);
assert.equal(legacy.mfu.computeActiveUs,30);assert.ok(legacy.mfu.computeRatio>0);
assert.equal(legacy.mfu.isE2E,true);assert.equal(legacy.observation.hasIssue,false);

const taskPattern=/DISPATCH|COMBINE|GMM[12]|SWIGLU/;
const passive=baseline.map(e=>taskPattern.test(e.name)?{...e,name:e.name.replace(/_0$/,'_ISSUE_0'),
  args:{...e.args,observation:'scalar_issue_with_native_dependencies',timer_protocol:'passive_aiv1_async_v2'}}:e);
passive.push(event('wave_0_AIV_DISPATCH_SYNC_0',6,2,'AIV'),event('wave_0_AIV_SWIGLU_SYNC_0',29,2,'AIV'),
  event('wave_0_AIC_GMM1_BEGIN_0',5,0.1),event('wave_0_AIC_GMM1_END_0',26,0.1),
  event('wave_global_AIV_QUANT_ISSUE_0',-5,4,'AIV'));
const result=analyze(passive);
for(const stage of ['gemm1','gemm2','dispatch','combine','swiglu']){
  assert.equal(result.stages[stage].count,2,stage);assert.equal(result.stages[stage].timingMode,'issue',stage);
  assert.equal(result.stages[stage].sumCoreUs,legacy.stages[stage].sumCoreUs,stage+'不得混入SYNC/标记');
}
assert.equal(result.mfu.computeRatio,null);assert.equal(result.mfu.computeEnvelopeRatio,null);
assert.match(result.mfu.computeReason,/ISSUE/);assert.ok(result.mfu.ratio>0);
assert.equal(result.inputQuantization.timeBasis,'original');assert.equal(result.inputQuantization.skippedIssueCount,1);
assert.equal(result.inputQuantization.removedWithinWindowUs,0);
assert.equal(result.events.find(e=>e.name.includes('DISPATCH_SYNC')).timing,'sync');
assert.equal(result.events.find(e=>e.name.includes('GMM1_BEGIN')).timing,'marker');
assert.equal(result.events.find(e=>e.name.includes('GMM1_END')).stage,null);
assert.equal(result.summary.dispatch.sumCoreUs,10);assert.equal(result.summary.dispatch.timingMode,'issue');
assert.equal(result.cores[0].dispatch.timingMode,'issue');assert.equal(result.cube.cores[0].timingMode,'issue');
for(const name of ['AIC_GMM1_WAIT_0','AIV_DISPATCH_SYNC_0','AIV_COMBINE_SYNC_0','AIC_GMM1_END_0',
  'AIV_GMM1_PROLOGUE_ISSUE_0','wave_global_AIC_SHARED_GMM1_END_0'])assert.equal(A.stageOf(name),null,name);
assert.equal(A.stageOf('wave_global_AIC_SHARED_GMM1_ISSUE_0'),'shared_gemm');
assert.equal(A.stageOf('wave_2_AIV_DISPATCH_CLOS_ISSUE_3'),'dispatch');
// 不依赖ISSUE后缀：明确的被动提交元数据也必须阻止仅计算MFU被误算。
const metadataOnly=analyze(baseline.map(e=>taskPattern.test(e.name)?{...e,args:{...e.args,observation:'scalar_issue_with_native_dependencies'}}:e));
assert.equal(metadataOnly.mfu.computeRatio,null);assert.equal(metadataOnly.stages.gemm1.timingMode,'issue');
const mixed=analyze([...passive,event('wave_1_AIC_GMM1_0',32,1)]);
assert.equal(mixed.stages.gemm1.timingMode,'mixed');assert.equal(mixed.mfu.computeRatio,null);
assert.ok(mixed.warnings.some(w=>w.includes('混合观测')));
// 旧QUANT扣除行为保留；ISSUE时则完全保留时间轴。
const quant=analyze([...baseline,event('wave_global_AIV_QUANT_0',-5,4,'AIV')]);
assert.equal(quant.inputQuantization.matchedEventCount,1);assert.equal(quant.inputQuantization.removedWallUs,4);
const clipped=A.analyze(A.parse(JSON.stringify(passive)),{...config,windowMode:'manual',windowStart:10,windowEnd:40});
assert.equal(clipped.mfu.ratio,null);assert.equal(clipped.volume.dispatch,null);
const compressed=analyze(passive.map(e=>({...e,args:{...e.args,input_sync_mode:'compress'}})));
assert.equal(compressed.mfu.ratio,null);assert.equal(compressed.cube.activeRatio,null);
// 单核缺失、跨Rank筛选和原始事件保留。
const missing=analyze(passive.filter(e=>e.args.group_id===0));
assert.equal(missing.mfu.ratio,null);assert.equal(missing.summary.dispatch.complete,false);
const rank1=baseline.map(e=>({...e,pid:'rank1',args:{...e.args,rank:1}}));
assert.equal(analyze([...passive,...rank1]).stages.dispatch.count,2);
assert.equal(passive[0].name,'AIV_INPUT_SYNC');
console.log('PASS: legacy, ISSUE/metadata, mixed, SYNC/markers, QUANT, clipped/compressed, missing cores, Rank0 isolation.');

// 可选真实回归只在本机读文件，不复制或上传用户Trace。
if(process.argv[2]){
  const text=fs.readFileSync(process.argv[2],'utf8'),t=performance.now();
  const r=A.analyze(A.parse(text,path.basename(process.argv[2]),{rank0Only:true}),{ep:32,totalExperts:384,bs:4096});
  for(const [stage,count] of Object.entries({gemm1:2304,gemm2:2688,dispatch:1280,combine:1280})){
    assert.equal(r.stages[stage].count,count,stage);assert.equal(r.stages[stage].coreCount,32,stage);
    assert.equal(r.stages[stage].timingMode,'issue',stage);
  }
  assert.equal(r.mfu.computeRatio,null);assert.equal(r.mfu.computeEnvelopeRatio,null);
  assert.equal(r.window.isE2E,false);assert.ok(r.mfu.ratio>0);
  assert.equal(r.inputQuantization.removedWithinWindowUs,0);
  assert.equal(r.allEvents.find(e=>e.name==='wave_0_AIC_GMM2_ISSUE_0').start,291.953);
  console.log(JSON.stringify({realTrace:true,events:r.allEvents.length,issueTasks:r.observation.issueCount,
    windowUs:r.window.durationUs,windowMfu:r.mfu.ratio,dispatchCoreMeanUs:r.summary.dispatch.meanCoreUs,
    combineCoreMeanUs:r.summary.combine.meanCoreUs,analysisMs:performance.now()-t},null,2));
}
