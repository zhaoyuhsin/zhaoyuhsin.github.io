# YuHsin · 算子性能工作台

个人网站及MegaMoE工具的静态发布仓库。站点文件在 `docs/`，不需要服务端或npm依赖。

计划地址：`https://zhaoyuhsin.github.io/`。仿真工具位于 `/megamoe/`，交互Wiki位于 `/megamoe/wiki.html#ratio-lab`。

## 本地查看

直接打开 `docs/index.html`，或在仓库目录运行 `python -m http.server 8765 --bind 127.0.0.1 --directory docs`，访问 `http://127.0.0.1:8765/`。

## GitHub Pages

仓库名使用 `zhaoyuhsin.github.io`，将本目录提交到该仓库的 `main` 分支。在仓库 **Settings → Pages** 中选择 **Deploy from a branch → main → /docs → Save**。

`docs/.nojekyll` 保留纯静态发布方式。后续提交到main会触发更新。仓库中不包含真实Trace文件、账号凭据、设备配置或原算子工程。

## 更新工具

首页源文件是 `docs/index.html`、`docs/assets/home.css` 和 `docs/assets/home.js`。工具目录来自已维护的离线网页：

```text
node scripts/build-tools.mjs <MegaMoe-Offline目录>
node scripts/check-site.mjs
```

脚本只复制明确列出的网页与依赖，不复制旧备份、测试产物、实际Trace或设备文件。首页的小实验复用Wiki内的计算引擎。网页配置保存在访问者浏览器中，Trace分析不上传文件。
