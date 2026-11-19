(function (root) {
  'use strict';
  const MIB = 1024 * 1024;
  const PIPE = ['MTE2', 'Vector', 'MTE3'];
  const profiles = {
    dq1: { task: 'dequant', outer: 7168, width: 256, block: 65536, inputSlots: 3, outputSlots: 2, read: 124.949, write: 209.428, bit: 760.257, reference: 10.844 },
    dq2: { task: 'dequant', outer: 3072, width: 256, block: 65536, inputSlots: 3, outputSlots: 2, read: 125.428, write: 207.174, bit: 751.847, reference: 5.095 },
    quant: { task: 'quant', outer: 4, width: 7168, cores: 64, inputSlots: 1, outputSlots: 1 },
    swiglu: { task: 'swiglu', outer: 8, width: 128, cores: 1, inputSlots: 1, outputSlots: 1 },
    unpermute: { task: 'unpermute', outer: 4, width: 7168, topk: 6, cores: 64, inputSlots: 2, outputSlots: 2 },
  };
  const hardware = {
    ghz: 1.65, read: 154, write: 196, readCap: 5300, writeCap: 4800,
    ubRead: 512, ubWrite: 256, ubMixed: 512, bit: 844.183,
    simpleCpi: 0.5, castCpi: 1, reduceCpi: 1, expCpi: 2, divCpi: 2,
    overhead: 0, cores: 32, topk: 6, block: 65536,
  };
  const ceil = Math.ceil;
  const align = (n, a) => ceil(n / a) * a;
  const peakTime = (n, gbps) => n / (1000 * gbps);

  function preset(id) {
    if (!Object.hasOwn(profiles, id)) throw new Error('未知预设');
    const { reference, ...settings } = profiles[id];
    return { id, ...hardware, ...settings };
  }

  function quantify(rows, width, h) {
    const groups = ceil(width / 32);
    const v256 = rows * ceil(width / 256);
    const v128 = rows * ceil(groups / 128);
    return [
      { name: '指数归约', simple: 3*v256, reduce: v256, read: v256*512, write: v256*16 },
      { name: '生成 scale', simple: 13*v128, read: rows*groups*2, write: rows*groups*3 },
      { name: '缩放、转换与打包', simple: 5*v256, cast: 8*v256, read: v256*528, write: v256*256 },
    ].map(part => vectorPart(part, h));
  }

  function vectorPart(part, h) {
    const cycles = (part.simple || 0)*h.simpleCpi + (part.cast || 0)*h.castCpi +
      (part.reduce || 0)*h.reduceCpi + (part.exp || 0)*h.expCpi + (part.div || 0)*h.divCpi;
    const ubCycles = Math.max(part.read/h.ubRead, part.write/h.ubWrite, (part.read+part.write)/h.ubMixed);
    return { ...part, issueUs: cycles/(h.ghz*1000), ubUs: ubCycles/(h.ghz*1000),
      serviceUs: Math.max(cycles, ubCycles)/(h.ghz*1000) };
  }

  function makeBlocks(h) {
    const width = h.width;
    if (h.task === 'dequant') {
      const weights = h.outer * width;
      if (width % 128 || h.outer % 64 || h.block % 256)
        throw new Error('位展开要求 N 对齐 128、K 对齐 64，块大小对齐 256 个权重');
      const ub = h.block*(h.inputSlots/2 + h.outputSlots);
      if (ub > 256*1024) throw new Error('FP4 输入槽与 FP8 输出槽合计超过 256 KiB UB');
      const blocks = Math.ceil(weights / h.block);
      if (blocks > 8192) throw new Error('单任务最多 8192 个流水块');
      const jobs = Array.from({length: blocks}, (_, i) => {
        const q = Math.min(h.block, weights-i*h.block);
        const bitUs = peakTime(3*q, h.bit);
        const ubUs = Math.max(q/2/h.ubRead, q/h.ubWrite, 1.5*q/h.ubMixed)/(1000*h.ghz);
        return { read: q/2, write: q, vector: [{name:'FP4 位展开', bitOps:3*q, issueUs:bitUs, ubUs, serviceUs:Math.max(bitUs, ubUs)}] };
      });
      return { jobs, active: h.cores, footprint: ub, work: [['权重元素 / 生产者', weights], ['8-bit 位操作 / 生产者', 3*weights]],
        scope: '每个 AIV0 处理一个完整 K 权重任务。AIC 搬运 B scale，此处计入 FP4 数据和 FP8 展开数据。' };
    }

    if (width % 128) throw new Error('向量预设要求宽度对齐到 128，匹配已建模的完整列块');
    const rowsPerCore = ceil(h.outer / h.cores);
    const active = ceil(h.outer / rowsPerCore);
    const rows = h.task === 'swiglu' ? h.outer : rowsPerCore;
    if (h.task === 'swiglu' && h.cores !== 1) throw new Error('SwiGLU 的输入是一块由单 AIV 处理的 tile');
    if (h.task === 'swiglu' && rows*width > 8192)
      throw new Error('SwiGLU 请输入单个子 tile，最多 8192 个输出元素');
    let job;
    let work;
    if (h.task === 'unpermute') {
      if (h.topk > 32) throw new Error('Unpermute 预设支持 topK ≤ 32');
      const loops = width/128;
      job = { read: 2*width*h.topk + 4*h.topk, write: 2*width, vector: [
        vectorPart({ name:'BF16 展开、加权归并、输出转换',
          simple: loops*(4*h.topk-2), cast: loops*(h.topk+3),
          read: 2*width*h.topk+4*h.topk*loops, write: 2*width }, h),
      ] };
      work = [['FP32 乘法 / 最忙核', width*h.topk*rows], ['FP32 加法 / 最忙核', width*(h.topk-1)*rows]];
    } else {
      const isActivation = h.task === 'swiglu';
      const taskRows = isActivation ? rows : 1;
      const n = taskRows*width;
      const vector = quantify(taskRows, width, h);
      if (isActivation) {
        const v64 = n/64;
        vector.unshift(vectorPart({name:'SwiGLU 含 clamp', simple:6*v64, cast:3*v64,
          exp:v64, div:v64, read:4*n, write:2*n}, h));
      }
      const packet = align(width,256) + align(ceil(width/32),32);
      job = {read:n*(isActivation ? 4:2),
        write:isActivation ? n+taskRows*ceil(width/32):packet, vector};
      work = [['量化元素 / 最忙核',width*rows], ['scale 组 / 最忙核',ceil(width/32)*rows]];
      if (isActivation) work.push(['Exp / Div 各自的元素数',n]);
    }
    // 非 DQ 预设以完整 token 或 tile 为块。槽位受输入输出各自的生命周期约束。
    const bytes = h.task==='unpermute' ? (job.read-4*h.topk)*h.inputSlots + align(4*h.topk*rows,32)
      : job.read*h.inputSlots + job.write*h.outputSlots;
    if (bytes > 256*1024)
      throw new Error('当前块的输入输出数据超过 256 KiB，缩小 tile 或槽数');
    return { jobs:Array(h.task === 'swiglu' ? 1 : rows).fill(job), active, work,
      scope:h.task === 'swiglu' ? '每任务一个 AIV1 tile。激活按 BF16 输入与 MXFP8 输出建模，量化使用通用指令账本近似。'
        : '按 token 连续分片，显示最忙 AIV。每块涵盖一个完整 token，依次读入、向量计算和写回。' };
  }

  function simulate(h) {
    if (!['dequant','quant','swiglu','unpermute'].includes(h.task)) throw new Error('未知任务类型');
    for (const key of ['outer','width','cores','topk','block','inputSlots','outputSlots']) {
      if (!Number.isSafeInteger(h[key]) || h[key] <= 0 || h[key] > 1048576)
        throw new Error(key + ' 必须为 1～1048576 的整数');
    }
    if (h.cores>64 || h.inputSlots>8 || h.outputSlots>8 || h.outer>8192)
      throw new Error('核数最多 64，单向槽数最多 8，行数或 K 最多 8192');
    if (h.task==='unpermute' && h.inputSlots!==h.outputSlots)
      throw new Error('Unpermute 原位输入与输出必须使用同一组槽');
    for (const key of ['ghz','read','write','ubRead','ubWrite','ubMixed','bit','simpleCpi','castCpi','reduceCpi','expCpi','divCpi']) {
      if (!Number.isFinite(h[key]) || h[key] <= 0) throw new Error(key+' 必须为正数');
    }
    for (const key of ['overhead','readCap','writeCap']) {
      if (!Number.isFinite(h[key]) || h[key] < 0) throw new Error(key+' 必须为非负数');
    }
    const {jobs, active, work, footprint, scope} = makeBlocks(h);
    const isDq = h.task === 'dequant';
    // rank 上限采用等分近似。DQ 的写端是 UB→L1，不占 GM 写聚合预算。
    const read = h.readCap ? Math.min(h.read, h.readCap/active) : h.read;
    const write = !isDq && h.writeCap ? Math.min(h.write,h.writeCap/active) : h.write;
    const ends = [[],[],[]];
    const events = [];
    const totals = [0,0,0];
    const vectorParts = new Map();
    let readB = 0;
    let writeB = 0;
    jobs.forEach((job, i) => {
      readB += job.read;
      writeB += job.write;
      const phases = [peakTime(job.read,read), job.vector.reduce((sum,p)=>sum+p.serviceUs,0), peakTime(job.write,write)];
      for (const part of job.vector) {
        const old = vectorParts.get(part.name) || { name:part.name, issueUs:0, ubUs:0, serviceUs:0 };
        for (const key of ['issueUs','ubUs','serviceUs']) old[key] += part[key];
        vectorParts.set(part.name,old);
      }
      const inputRelease = h.task==='unpermute' ? ends[2] : ends[1];
      const inputReady = i>=h.inputSlots ? inputRelease[i-h.inputSlots] : 0;
      const outputReady = i>=h.outputSlots ? ends[2][i-h.outputSlots] : 0;
      const starts = [Math.max(ends[0][i-1]||0,inputReady)];
      starts.push(Math.max(starts[0]+phases[0],ends[1][i-1]||0,outputReady));
      starts.push(Math.max(starts[1]+phases[1],ends[2][i-1]||0));
      phases.forEach((duration, j) => {
        const end = starts[j]+duration;
        ends[j].push(end);
        totals[j] += duration;
        events.push({block:i+1, pipe:PIPE[j], startUs:starts[j], endUs:end});
      });
    });
    const reference = Object.hasOwn(profiles,h.id) ? profiles[h.id].reference : undefined;
    const base = reference === undefined ? null : preset(h.id);
    const matchesReference = base && Object.keys(base).every(k => k==='overhead' || h[k]===base[k]);
    const longest = Math.max(...totals);
    return { modelVersion:1, task:h.task, scope, active, blocks:jobs.length, footprintBytes:footprint, work,
      readBytes:readB,writeBytes:writeB, effectiveReadGBps:read,effectiveWriteGBps:write,
      services:PIPE.map((pipe,i)=>({pipe,us:totals[i]})), vectorParts:[...vectorParts.values()],
      boundPipe:PIPE[totals.indexOf(longest)], serviceFloorUs:longest, scheduleUs:ends[2].at(-1),
      taskUs:ends[2].at(-1)+h.overhead, serialUs:totals.reduce((a,b)=>a+b,0),
      historicalTaskUs:matchesReference ? reference : null, events };
  }

  const api = { preset, simulate };
  if (typeof module==='object' && module.exports) module.exports=api;
  else root.AivModel=api;
})(globalThis);
