import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=process.argv[2];
if(!source)throw Error('请指定网页目录：node scripts/build-tools.mjs <MegaMoe-Offline目录>');
const files=['index.html','benchmark.html','wiki.html','wiki-standalone.html','memory.html','mfu-standalone.html','trace-v2.html','trace.html','trace-standalone.html','catlass-wave.js','catlass-network.js','gemm-memory.js','memory-model.js','memory-page.js','trace-analysis.js','trace-page.js','memory-audit.md'];
for(const file of files)if(!fs.statSync(path.join(source,file)).isFile())throw Error('缺少工具文件：'+file);
const destination=path.join(root,'megamoe');fs.mkdirSync(destination,{recursive:true});
const nav='<div style="padding:9px 20px;background:#f6f7f2;color:#245c48;font:12px/1.6 system-ui,sans-serif;display:flex;gap:20px;flex-wrap:wrap"><a style="color:inherit" href="../index.html">← YuHsin 个人主页</a><a style="color:inherit" href="index.html">流水仿真</a><a style="color:inherit" href="wiki.html#ratio-lab">通算比例</a><a style="color:inherit" href="trace-v2.html">Trace分析</a></div>';
const manifest=[];
for(const file of files){
  let content=fs.readFileSync(path.join(source,file),'utf8');
  // 独立分享文件保留原状；普通在线工具加上回到个人主页的导航。
  if(file.endsWith('.html')&&!file.includes('standalone'))content=content.replace('<body>','<body>\n'+nav);
  if(file==='memory-audit.md')content=content.replace(/源工作树[^。]*。/g,'源码以所标注提交为参考。');
  fs.writeFileSync(path.join(destination,file),content);
  manifest.push({file,sha256:crypto.createHash('sha256').update(content).digest('hex')});
}
// 首页小实验复用Wiki原计算引擎，避免另写一套精度或带宽公式。
const wiki=fs.readFileSync(path.join(source,'wiki.html'),'utf8');
const engine=[...wiki.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).find(code=>code.includes('root.MegaMoeRatio='));
if(!engine)throw Error('Wiki中缺少通算比例计算引擎');
fs.writeFileSync(path.join(root,'assets','ratio-model.js'),engine);
fs.writeFileSync(path.join(destination,'build-info.json'),JSON.stringify({source:'MegaMoe-Offline',files:manifest},null,2));
console.log(`Prepared ${files.length} tool files and the shared ratio engine.`);
