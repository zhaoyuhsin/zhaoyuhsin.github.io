(function () {
  'use strict';
  // 容量与流量分开：固定槽位按最大形状预留，主数据按16行对齐计算。
  function model(M, N, K, mode = 'auto') {
    if (!Number.isInteger(M) || M < 1 || M > 256) throw Error('M 必须是1～256的整数；多 M-block 请分别建模。');
    for (const [name, value] of [['N', N], ['K', K]])
      if (!Number.isInteger(value) || value < 256 || value > 65536 || value % 256) throw Error(name + ' 必须是256的倍数，范围256～65536。');
    if (!['auto', '128', '256'].includes(mode)) throw Error('未知 K 分块策略。');
    const m = M <= 128 ? 128 : 256, rows = Math.ceil(M / 16) * 16;
    const k = mode === 'auto' ? (m === 128 ? 256 : 128) : Number(mode);
    const n = 32768 / k, parts = 256 / n, tiles = N / 256;
    const a = 2 * m * k, b = 2 * n * k, c = m * 256 * 4;
    return {M, N, K, m, rows, k, n, parts, tiles, a, b, c,
      fits: a <= 65536 && b <= 65536 && c <= 262144,
      l1A: 3 * m * (2 * k), l1B: 4 * n * (2 * k), l1SF: 65536,
      kRounds: K / k, l1Rounds: K / (2 * k),
      mad: tiles * parts * K / k,
      aLoads: tiles * K / k, bLoads: tiles * parts * K / k,
      aBytes: rows * K * tiles, bBytes: N * K,
      packedBytes: N * K / 2, outputBytes: M * N * 2,
      usefulFlops: 2 * M * N * K};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {model};
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id), kib = x => (x / 1024).toLocaleString('zh-CN') + ' KiB';
  function render() {
    try {
      const r = model(Number($('gm-m').value), Number($('gm-n').value), Number($('gm-k').value), $('gm-mode').value);
      $('gm-error').hidden = true;
      const allocations = [['L0A', r.a, 65536, `2槽 × ${r.m}×${r.k} FP8`],
        ['L0B', r.b, 65536, `2槽 × ${r.n}×${r.k} FP8`],
        ['L0C', r.c, 262144, `${r.parts}块 × ${r.m}×${r.n} FP32（同一任务的N分区）`],
        ['L1', r.l1A + r.l1B + r.l1SF, 524288, `A三槽 ${kib(r.l1A)} + B四槽 ${kib(r.l1B)} + scale ${kib(r.l1SF)}`]];
      let content = `<p><b>有效 M=${r.M}，对齐行数=${r.rows}；MAD 上限 ${r.m}×${r.n}×${r.k}。</b>完整输出分为 ${r.tiles} 个 N256任务；每个任务沿 N 分成 ${r.parts} 块，沿 K 累加 ${r.kRounds} 轮。L1 的 K 分块为 ${2*r.k}。</p>`;
      content += '<div class="scroll"><table><tr><th>存储</th><th>预留 / 容量</th><th>分区</th></tr>' + allocations.map(([name, used, cap, desc]) => `<tr><td>${name}</td><td>${kib(used)} / ${kib(cap)} ${used > cap ? '⚠ 溢出' : ''}</td><td>${desc}<div class="track"><span class="seg" style="width:${Math.min(100, used/cap*100)}%">${(used/cap*100).toFixed(0)}%</span></div></td></tr>`).join('') + '</table></div>';
      if (!r.fits) content += '<p class="notice">此形状不可用：L0A 无法容纳双缓冲；按原 L1 槽位数也会溢出。不能直接把大 M 的 K 增至256而保持其他设置。</p>';
      content += `<p>L0C 的 ${r.parts} 块不是跨任务乒乓：${r.parts === 2 ? '前128列与后128列各自保留部分和，复用同一份 L0A；两块都完成后才得到完整N256输出。' : '同一块保存全部256列的部分和。'}容量空余不自动变成可用的流水槽。</p>`;
      content += '<h3>固定同一 M/N/K：K128 与 K256 对照</h3><div class="scroll"><table><tr><th>口径（主数据，不含scale）</th><th>K128 / N256</th><th>K256 / N128</th></tr>';
      const variants = ['128', '256'].map(mode => model(r.M, r.N, r.K, mode));
      const rows = [
        ['容量是否合法', x => x.fits ? '可容纳（不等于源码选用）' : '溢出，不可执行'],
        ['每个N256任务：L1 K批次', x => x.l1Rounds],
        ['每个N分区：L0 K迭代', x => x.kRounds],
        ['全部任务：MAD次数', x => x.mad],
        ['全部任务：A装入L0次数', x => x.aLoads],
        ['全部任务：B装入L0次数', x => x.bLoads],
        ['A：GM→L1 / L1→L0 各自载荷', x => kib(x.aBytes)],
        ['B：GM→UB 的FP4载荷', x => kib(x.packedBytes)],
        ['B：UB→L1 / L1→L0 各自FP8载荷', x => kib(x.bBytes)],
        ['最终 BF16 输出（有效行）', x => kib(x.outputBytes)]
      ];
      content += rows.map(([label, value]) => `<tr><td>${label}</td>${variants.map(x => `<td>${value(x)}</td>`).join('')}</tr>`).join('') + '</table></div>';
      content += '<p>上表次数是主数据分块装载次数，不包含 scale、布局转换和事件指令。假设每个N256任务独立加载A，任务内部两个N128分区复用A；无跨任务缓存复用。K翻倍后A装载次数减半，但每次更多字节；B的N减半抵消K翻倍，MAD和B装载次数保持不变。溢出列仅供数学对照，不能当作可执行性能结果。</p>';
      $('gm-result').innerHTML = content;
    } catch (error) {
      $('gm-error').hidden = false; $('gm-error').textContent = error.message; $('gm-result').innerHTML = '';
    }
  }
  for (const id of ['gm-m', 'gm-n', 'gm-k', 'gm-mode']) $(id).addEventListener('change', render);
  render();
})();
