# YuHsin · 算子性能工作台

个人网站及MegaMoE工具的静态发布仓库。站点入口是根目录 `index.html`，不需要服务端或npm依赖。

访问地址：`https://zhaoyuhsin.github.io/`。仿真工具位于 `/megamoe/`，交互Wiki位于 `/megamoe/wiki.html#ratio-lab`。

## 本地查看

直接打开 `index.html`，或在仓库目录运行 `python -m http.server 8765 --bind 127.0.0.1`，访问 `http://127.0.0.1:8765/`。

## GitHub Pages

仓库名使用 `zhaoyuhsin.github.io`，将本目录提交到该仓库的 `main` 分支。在仓库 **Settings → Pages** 中选择 **Deploy from a branch → main → /(root) → Save**。

根目录 `.nojekyll` 保留纯静态发布方式。后续提交到main会触发更新。仓库中不包含真实Trace文件、账号凭据、设备配置或原算子工程。

## 更新工具

首页源文件是 `index.html`、`assets/home.css` 和 `assets/home.js`。工具目录来自已维护的离线网页：

```text
node scripts/build-tools.mjs <MegaMoe-Offline目录>
node scripts/check-site.mjs
```

脚本只复制明确列出的网页与依赖，不复制旧备份、测试产物、实际Trace或设备文件。首页的小实验复用Wiki内的计算引擎。网页配置保存在访问者浏览器中，Trace分析不上传文件。

## Trace V2.8：被动 timer 兼容

`megamoe/trace-v2.html` 仍是单文件离线页。支持 `*_ISSUE`（包括 wave 前缀）及明确的被动提交元数据；原事件名、Wave、专家与时间保留。WAIT/SYNC、BEGIN/END不计入GMM或通信任务，逐核流水和悬停仍可查看。

ISSUE表示Scalar提交，不表示硬件完成。阶段与核时统计标明提交口径，通信GB/s仅为字节/提交区间折算；含ISSUE的GEMM不显示仅计算MFU。不自动扣QUANT或压缩被动时间轴。缺少完整端点时，MFU只标为观测窗口代理。旧格式继续保留原口径。

回归检查：`node scripts/check-trace-issue.mjs`。可在命令后传本地EP32回归JSON路径；测试只读取，不复制或上传真实Trace。

## MegaMoE 性能实验室

公开页面：`https://zhaoyuhsin.github.io/megamoe-performance/index.html#ds-converge`。

包含性能、项目状态、调优 SKILL 三个界面；性能数据嵌入 HTML，不依赖本地文件或服务端。当前快照为 `ds-converge-final-1021`，旧冻结曲线与最新候选分开显示。原始 timer、CSV、日志、源码载荷和二进制不随公开页发布，页面对应入口明确标为本地归档。

更新时从保留完整证据的离线看板生成公开副本：

```text
node scripts/build-performance.mjs <离线看板目录>
node scripts/check-performance.mjs <离线看板目录>
node scripts/check-site.mjs
```

构建器不修改性能数值；隐藏机器绝对路径和地址，只发布明确列出的摘要文件。数值一致性、Markdown SHA、脚本语法、三界面数据及凭据特征由检查脚本验证。`publication.json` 记录输入和公开 HTML 的 SHA256。原始离线看板不被修改。
