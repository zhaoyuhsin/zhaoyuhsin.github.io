import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

// 只发布摘要；原始流水/CSV、机器路径和登录信息不属于公开页面。
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(process.argv[2] || '');
if (!process.argv[2] || !fs.existsSync(path.join(source, 'results.json'))) {
  throw Error('Usage: node scripts/build-performance.mjs <offline-dashboard-directory>');
}
const out = path.join(root, 'megamoe-performance');
const sha = text => crypto.createHash('sha256').update(text).digest('hex');
const originalHtml = fs.readFileSync(path.join(source, 'index.html'), 'utf8');
const original = JSON.parse(fs.readFileSync(path.join(source, 'results.json'), 'utf8'));
const snapshot = original.stage1?.r31_ep32_performance_resume_1024 ? 'r31-ep32-resume-1024' : original.stage1?.r31_ep32_performance_1024 ? 'r31-ep32-performance-1024' : original.stage1?.five_rounds_1024 ? 'five-rounds-1024' : original.stage1?.r31_aiv1_wave_matrix ? 'r31-aiv1-wave-matrix-1024' : original.stage1?.r31_two_round_wave ? 'r31-two-round-wave-1023' : original.stage1?.version_matrix ? 'version-matrix-1023' : original.stage1?.ep32_r31 ? 'ep32-r31-layout-ds-1021-' + original.stage1.ep32_r31.status : 'ds-converge-final-1021';
const privateAnchor = '#private-artifacts';
let hiddenLinks = 0, hiddenPaths = 0;
function scrub(value) {
  if (typeof value === 'string') {
    if (value.startsWith('evidence/')) { hiddenLinks++; return privateAnchor; }
    return value.replace(/(?:[A-Z]:[\\/]|\/(?:home|root|opt|miniconda|tmp|workspace)\/)[^\s"'`<>，。；：、）)\]}]+/gi,
      () => { hiddenPaths++; return '[机器本地路径未公开]'; })
      .replace(/柜\d+\s*CPU\d+\s*[·/]?\s*/g, '')
      .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, '[地址未公开]');
  }
  if (Array.isArray(value)) return value.map(scrub);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => {
    if (/^(?:password|passwd|credential|credentials|secret|access_token|api_key|private_key)$/i.test(k)) {
      throw Error('Refusing credential-bearing field: ' + k);
    }
    return [k, scrub(v)];
  }));
  return value;
}
const data = scrub(original);
// 二进制/JIT文件清单不用于页面计算；公开保留软件版本及实验内已有的身份哈希。
if (data.environment) delete data.environment.files;
data.publication = {
  audience: 'public', snapshot,
  original_artifacts: 'local_archive_not_published',
  note: '数值与离线快照一致；原始timer/CSV及服务器文件未公开。',
};
const json = value => JSON.stringify(value).replace(/</g, '\\u003c');
const docsMatch = originalHtml.match(/<script[^>]*id="project-documents"[^>]*>([\s\S]*?)<\/script>/);
if (!docsMatch) throw Error('Missing embedded project documents');
const docs = JSON.parse(docsMatch[1]).map(doc => {
  const markdown = scrub(doc.markdown), html = scrub(doc.html);
  return {...doc, markdown, html, sha256: sha(markdown)};
});
function replaceOnce(text, needle, replacement) {
  if (!text.includes(needle) || text.indexOf(needle) !== text.lastIndexOf(needle)) {
    throw Error('Dashboard source changed; expected one known publication boundary: ' + needle.slice(0, 80));
  }
  return text.replace(needle, replacement);
}
let html = originalHtml.replace(/(<script[^>]*id="benchmark-data"[^>]*>)[\s\S]*?(<\/script>)/,
  (_, start, end) => start + json(data) + end);
html = html.replace(/(<script[^>]*id="project-documents"[^>]*>)[\s\S]*?(<\/script>)/,
  (_, start, end) => start + json(docs) + end);
html = replaceOnce(html, '<title>MegaMoE · 离线性能实验室</title>', '<title>MegaMoE 性能实验室 · YuHsin</title>');
html = html.replace('无需网络或服务器。', '公开性能快照、项目状态与调优方法。');
html = replaceOnce(html, '<body>', '<body>\n<aside id="private-artifacts" style="padding:12px 20px;background:#fff5dc;border-bottom:1px solid #e7cfa0;color:#5e491c;overflow-wrap:anywhere"><a href="../">← 返回个人网站</a> · 公开快照 ' + snapshot + '：性能、实验记录、项目状态及调优 SKILL 均保留。原始 timer/CSV 仅在本地归档，未公开；对应入口会标注“本地归档”，不是在线下载。</aside>');
// 静态占位保留跨页面锚点；运行时仍由原有renderDsConverge管理同一容器。
html = replaceOnce(html, '<div id="stage1-experiments"></div></section>', '<div id="stage1-experiments"></div></section><div id="ds-converge"></div>');
html = replaceOnce(html, '需要打开原始证据时，请连同同级evidence目录一起复制。', '公开页不发布原始附件；如需原始timer/CSV，请使用本地离线归档。');
const oldSafe = "function safeLink(url){return typeof url==='string'&&/^evidence\\/[A-Za-z0-9_./-]+$/.test(url)&&!url.split('/').includes('..')?url:'#evidence';}";
html = replaceOnce(html, oldSafe, `function safeLink(url){return ${JSON.stringify(privateAnchor)};}`);
// 动态渲染/切换统计/导入后均标注私有附件，不能伪装成公开可下载的证据。
html = replaceOnce(html, 'function render(){', `function markPrivateArtifacts(){
 document.querySelectorAll('a[href="#private-artifacts"]').forEach(a=>{
  if(!a.dataset.localArchive){a.dataset.localArchive='true';a.textContent+=' · 本地归档（未公开）';}
  a.title='原始附件保留在本地；公开页只提供性能摘要与实验说明。';
 });
}
function render(){`);
html = replaceOnce(html, "window.addEventListener('hashchange',route);", "window.addEventListener('hashchange',route);\nnew MutationObserver(markPrivateArtifacts).observe(document.body,{childList:true,subtree:true});");
html = replaceOnce(html, 'try{render();renderDocuments();route();}', 'try{render();renderDocuments();route();markPrivateArtifacts();}');

fs.mkdirSync(out, {recursive:true});
fs.writeFileSync(path.join(out, 'index.html'), html);
fs.writeFileSync(path.join(out, 'results.json'), JSON.stringify(data, null, 2) + '\n');
for (const doc of docs) fs.writeFileSync(path.join(out, doc.filename), doc.markdown);
fs.writeFileSync(path.join(out, 'publication.json'), JSON.stringify({
  snapshot: data.publication.snapshot, source_html_sha256: sha(originalHtml),
  source_results_sha256: sha(fs.readFileSync(path.join(source, 'results.json'))),
  public_html_sha256: sha(html), hidden_links: hiddenLinks, hidden_paths: hiddenPaths,
  artifacts_published: false, measurements_modified: false,
}, null, 2) + '\n');
console.log(JSON.stringify({output:'megamoe-performance/', html_bytes:Buffer.byteLength(html), hidden_links:hiddenLinks, hidden_paths:hiddenPaths}));
