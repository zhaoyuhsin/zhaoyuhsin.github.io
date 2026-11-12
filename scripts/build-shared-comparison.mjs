import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

// 只导出已确认的聚合数值；不复制原始流水、日志、节点或本机路径。
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=process.argv[2];
if(!source)throw Error('Usage: node scripts/build-shared-comparison.mjs <local-outputs-directory>');
const read=relative=>JSON.parse(fs.readFileSync(path.join(source,relative),'utf8'));
const sha=relative=>crypto.createHash('sha256').update(fs.readFileSync(path.join(source,relative))).digest('hex');
const currentFile='performance-compare-20261111/summary.json';
const current=read(currentFile);
assert.equal(current.status,'audited_complete');
const bs=[1,4,8,16,32,96,256,512,1024,2048,4096];
const historicalFiles={8:'wave-cv-vs-ds-1024/ep8-r01-8-a/summary.json',32:'wave-cv-vs-ds-1024/ep32-r05-1234-c/summary.json'};
const slowdown=(ours,ds)=>100*(ours/ds-1);
const shared=[8,32].map(ep=>{
  const line=current.lines.find(v=>v.ep===ep&&v.shared_experts_per_rank===1);
  assert.equal(line.status,'audited_complete');
  const rows=bs.map(batch=>{
    const get=method=>{
      const row=line.aggregates.find(v=>v.bs===batch&&v.method===method&&v.metric==='full_fused_kernel_duration');
      assert.equal(row.independent_starts,2);
      assert.equal(row.all_sessions_complete,true);
      return {rank_min_us:row.rank_min_us,rank_max_us:row.rank_max_us,rank_min_run_range_us:row.rank_min_run_range_us};
    };
    const ours=get('main_latest'),layout_ds=get('ds_layout');
    return {bs:batch,ours,layout_ds,slowdown_pct:slowdown(ours.rank_min_us,layout_ds.rank_min_us)};
  });
  return {ep,rows};
});
const historical=[8,32].map(ep=>{
  const report=read(historicalFiles[ep]);
  assert.equal(report.status,'audited_complete');
  assert.equal(report.packages.combined.revision,'4a0de8571b18d4dde6e55e03e895ea9f55cc0e1c');
  const rows=report.cases.map(c=>{
    assert.equal(c.status,'audited_complete');
    const get=method=>{
      const m=c.methods[method];assert.equal(m.starts,2);
      return {rank_min_us:m.mean_us,rank_max_us:m.rank_max_mean_us,
        rank_min_run_range_us:[Math.min(...m.runs.map(r=>r.mean_us)),Math.max(...m.runs.map(r=>r.mean_us))]};
    };
    const ours=get('combined'),layout_ds=get('control');
    return {bs:c.bs,ours,layout_ds,slowdown_pct:slowdown(ours.rank_min_us,layout_ds.rank_min_us)};
  });
  assert.deepEqual(rows.map(v=>v.bs),[32,96,256,512,1024,2048,4096]);
  return {ep,rows};
});
const data={schema_version:1,published_on:'2026-11-12',title:'MegaMoE vs 布局优化 DS：共享专家对照',
  model:{hidden:7168,intermediate:3072,experts:384,topk:6,bs_definition:'per_rank',seed:20260922},
  aggregation:'20 rounds per start, discard first 10; mean of per-round rank-min, then mean of two independent starts',
  slowdown_definition:'100 * (ours / layout_ds - 1); positive is slower, negative is faster',
  accuracy:'not_run_user_requested',raw_artifacts_published:false,
  current_shared:{shared_experts_per_rank:1,shared_dtype:'FP8 W8A8 (E4M3 / E8M0 scales)',routed_dtype:'W4A8',
    metric:'full_fused_kernel_duration',timer:'all measured invocations, all ranks',
    ours_revision:current.frozen_versions.main_latest.revision,layout_ds_revision:current.frozen_versions.ds_layout.revision,
    main_promoted:false,cohort:'same rack; methods use the same ordered node cohort within each EP',results:shared},
  historical_no_shared:{historical:true,shared_experts_per_rank:0,metric:'full_fused_kernel_duration',timer:false,
    trace:'separate rank0 timer pass',ours_revision:'4a0de8571b18d4dde6e55e03e895ea9f55cc0e1c',
    layout_ds_revision:current.frozen_versions.ds_layout.revision,cohort:'each EP uses a matched cohort; EP8 and EP32 are independent historical windows',results:historical},
  limitations:['Current no-shared MegaMoE vs layout-DS matched measurement is missing; the current no-shared reference was the eager small-operator cascade.',
    'Historical no-shared and current shared rows use different candidates, windows and timer settings; do not subtract them to infer shared-expert overhead.',
    'Rank-min is not slowest-rank latency. Rank-max and independent-start ranges are retained in this JSON.',
    'Timer-on overhead was not independently measured for the shared cases; no universal overhead bound is claimed.',
    'Main applies routed weights after GMM2; DS applies them before GMM2. This performance-only comparison does not establish output equivalence.',
    'Source and loaded extensions were audited; the DS JIT cache binary was not independently audited.'],
  evidence:[{role:'current shared audit aggregate',sha256:sha(currentFile)},
    ...[8,32].map(ep=>({role:`historical no-shared EP${ep} audit aggregate`,sha256:sha(historicalFiles[ep])}))]};
const num=v=>v.toFixed(2);
const pct=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(2)+'%';
function table(lines,id,caption){
  const batches=lines[0].rows.map(r=>r.bs);
  return `<div class="table-scroll" role="region" aria-label="${caption}" tabindex="0"><table id="${id}"><caption>${caption}</caption><thead><tr><th rowspan="2" scope="col">BS / 卡</th><th colspan="3" scope="colgroup">EP8</th><th colspan="3" scope="colgroup">EP32</th></tr><tr>${[8,32].map(()=>'<th scope="col">我们 μs</th><th scope="col">布局DS μs</th><th scope="col">我们较DS慢</th>').join('')}</tr></thead><tbody>${batches.map(batch=>`<tr><th scope="row">${batch}</th>${[8,32].map(ep=>{
    const r=lines.find(l=>l.ep===ep).rows.find(v=>v.bs===batch);
    const state=Math.abs(r.slowdown_pct)<1?'near':r.slowdown_pct>0?'slower':'faster';
    return `<td>${num(r.ours.rank_min_us)}</td><td>${num(r.layout_ds.rank_min_us)}</td><td class="${state}">${pct(r.slowdown_pct)}</td>`;
  }).join('')}</tr>`).join('\n')}</tbody></table></div>`;
}
const html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>共享专家性能对照 · MegaMoE · YuHsin</title><meta name="description" content="EP8与EP32的MegaMoE对布局优化DS性能：FP8共享专家双轮实测、历史无共享专家对照与计时边界。">
<link rel="icon" href="../assets/favicon.svg"><link rel="stylesheet" href="../assets/home.css">
<style>
main{max-width:1180px;padding-bottom:60px}.page-hero{padding:48px 0 25px}.page-hero h1{font-size:clamp(28px,4vw,44px);margin-bottom:14px}.page-hero p{max-width:850px;color:var(--muted)}.page-hero .eyebrow{margin-bottom:16px}.anchors{display:flex;gap:12px 24px;flex-wrap:wrap;margin:22px 0;font-size:13px;color:var(--accent)}section{scroll-margin-top:20px}.block{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;margin:22px 0}.block h2{font-size:23px;margin:0 0 12px;font-weight:600}.block p,.block li{font-size:14px}.note{background:var(--accent-light);border-left:3px solid var(--accent);padding:13px 16px;font-size:13px}.warning{background:#fff5e7;border-left-color:#ad6a28}.meta{font-size:12px!important;color:var(--muted);line-height:1.8}.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:7px;margin:20px 0 10px}table{width:100%;min-width:760px;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}caption{text-align:left;padding:12px 14px;font-weight:600;white-space:normal}th,td{text-align:right;padding:10px 13px;border-top:1px solid var(--line)}thead th{background:#eef3ee;font-size:12px}tbody th{text-align:left;background:#f8faf6}tbody tr:hover td{background:#f6f8f3}.slower{color:#9d342c;font-weight:650}.faster{color:#196649;font-weight:650}.near{color:var(--muted)}.strategy{min-width:700px;white-space:normal}.strategy td,.strategy th{text-align:left;vertical-align:top}.strategy td{width:40%}code{font-size:12px;overflow-wrap:anywhere}.hash{word-break:break-all}.foot{font-size:12px;color:var(--muted)}@media(max-width:700px){.site-header nav{gap:12px}.block{padding:17px}.page-hero{padding-top:30px}.block h2{font-size:20px}}
</style></head><body><a class="skip" href="#main">跳到正文</a>
<header class="site-header"><a class="brand" href="../"><span class="mark">YH</span>YuHsin</a><nav aria-label="主导航"><a href="index.html">性能实验室</a><a href="../#experiments">全部实验</a></nav></header>
<main id="main"><header class="page-hero"><p class="eyebrow">MEGAMOE / MATCHED PERFORMANCE · 2026-11-12</p><h1>共享专家，<em>收益与代价一起看。</em></h1><p>我们与布局优化 DS 的 EP8 / EP32 耗时对照。共享专家采用 FP8 W8A8，路由专家采用 W4A8；每个已列配置均有两次独立启动，保留全部回退项。</p><nav class="anchors" aria-label="本页内容"><a href="#shared">1 个共享专家</a><a href="#no-shared">历史无共享专家</a><a href="#strategy">实现策略</a><a href="#protocol">协议与数据</a></nav><div class="note"><strong>劣化 =（我们 ÷ 布局DS − 1）× 100%。</strong>正数表示我们更慢，负数表示我们更快。灰色表示绝对差异小于 1%，不是统计显著性判断。</div></header>
<section class="block" id="shared"><h2>1 个共享专家 · 本次完整对照</h2><p class="meta">完整融合 kernel Duration · timer-on：全部 rank、全部测量调用 · 单位 μs · 共享 FP8 E4M3 / E8M0 scale · 每 rank 1 个同尺寸共享专家</p>
${table(shared,'shared-table','2026-11-11 实测：共享专家 FP8 W8A8，越低越好')}
<p>EP8 各 BS 均慢于布局 DS；EP32 在 BS4–256 快约 3.31%–14.27%，BS512 慢约 5.77%，其余按表保留小幅胜负。没有按 shape 拼接不同候选的最低值。</p><p class="meta">我们：c5d68f0d + shared-fp8-1110 隔离候选，未自动合入 main。布局DS：d0a77451。当前共享专家缺少配套 timer-off 测量，不能假设 timer 开销恒小于 1%。</p></section>
<section class="block" id="no-shared"><h2>无共享专家 · 历史参考，不是本次同版本消融</h2><div class="note warning"><strong>当前版本的无共享专家 vs 布局 DS 对照缺测。</strong>本次无共享专家的参考是小算子拼接，不是 DS。下表为旧候选 4a0de857 的独立历史对照，不与上表相减推导共享专家额外开销。</div><p class="meta">完整通信 kernel Duration · 主性能 timer-off，另采 rank0 timer 流水 · 两次独立启动 · EP8 / EP32 各自为匹配窗口，两个 EP 不在同一历史节点组</p>
${table(historical,'no-shared-table','历史双轮实测：旧候选 4a0de857 vs 布局 DS d0a77451')}
<p class="meta">BS1 / 4 / 8 / 16 没有此历史配对结果，未补数。候选、窗口和 timer 条件不同，禁止与上表组成有 / 无共享的因果收益。</p></section>
<section class="block" id="strategy"><h2>共享路径的策略差异</h2><p>已核对本次冻结源码和 DS 头文件身份。以下是实现差异，不是各项耗时的独立归因。</p><div class="table-scroll" role="region" aria-label="共享路径实现差异" tabindex="0"><table class="strategy"><thead><tr><th>项目</th><th>我们</th><th>布局优化 DS</th></tr></thead><tbody>
<tr><th>调度顺序</th><td>共享 GMM1 → 路由专家计算 → 共享 GMM2。</td><td>调度器优先共享 GMM1、共享 GMM2，再调度路由专家任务。</td></tr>
<tr><th>通信重叠机会</th><td>共享 GMM1 与 Dispatch、共享 GMM2 与末尾 Combine 可部分重叠。</td><td>两层共享计算有机会覆盖等待路由元数据 / Dispatch 准备的时间；不保证所有 BS 更快。</td></tr>
<tr><th>FP8 共享权重供数</th><td>AIV0：GM → UB → L1；不解包，但占用 AIV0 搬运资源。</td><td>AIC 侧 MTE2：GM → L1；共享权重不经过 AIV0 UB。</td></tr>
<tr><th>小 M 计算</th><td>共享路径 M=256 tile。</td><td>共享 token 数 ≤128 时选择 M=128，否则 M=256。</td></tr>
<tr><th>最终输出</th><td>尾部检查共享激活就绪、做共享 GMM2 后合并。</td><td>共享 GMM2 提前产出，归约阶段检查就绪再合并。</td></tr></tbody></table></div><p class="meta">共享 GMM2 的位置、AIV0 供数与小 M 粒度是下一步消融候选。没有带宽计数器或同条件消融，不能把全部差距直接归因到其中一项。</p></section>
<section class="block" id="protocol"><h2>测量协议与边界</h2><ul><li>模型 H=7168、I=3072、E=384、TopK=6；BS 为每卡 / 每 rank BS；seed=20260922，均衡路由。</li><li>每次独立启动 warmup 10 次，采样 20 轮取后 10 轮；逐轮 rank-min 后求均值，再对两次启动求均值。不是全局最慢 rank 时延；JSON 同时保留 rank-max 和两次启动范围。</li><li>当前共享与历史无共享的版本、运行窗口、timer 条件不同，不能相减。timer-on 的软件阶段跨度不是纯计算忙时，阶段区间也不能相加为完整耗时。</li><li>本轮按用户要求只测性能：accuracy=not_run_user_requested。主路径在 GMM2 后应用路由权重，DS 在 GMM2 前应用；没有输出等价或生产验收声明。</li><li>原始 timer / CSV / 日志及服务器信息不公开；公开数值直接来自已审计聚合，不重算或重跑设备实验。DS JIT 缓存二进制尚未独立审计。</li></ul><p><a class="text-link" href="shared-expert-comparison.json" download>下载公开聚合 JSON（含 rank-max、运行范围及证据 SHA-256） ↓</a></p><p class="meta">证据身份：${data.evidence.map(e=>`${e.role}：<code class="hash">${e.sha256}</code>`).join('<br>')}</p></section>
<p class="foot">本页发布于 2026-11-12。数值来自本地已审计记录；不替换性能实验室现有历史曲线，也不代表最新 main 已完成共享专家生产验收。</p></main><footer class="site-footer"><a href="../">← 返回个人网站</a><a href="index.html">性能实验室 ↗</a></footer></body></html>`;
const out=path.join(root,'megamoe-performance');
fs.writeFileSync(path.join(out,'shared-expert-comparison.json'),JSON.stringify(data,null,2)+'\n');
fs.writeFileSync(path.join(out,'shared-expert-comparison.html'),html+'\n');
console.log(JSON.stringify({shared_rows:shared.reduce((n,v)=>n+v.rows.length,0),historical_rows:historical.reduce((n,v)=>n+v.rows.length,0),raw_artifacts_published:false}));
