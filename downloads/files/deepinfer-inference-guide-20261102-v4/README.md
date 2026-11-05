# DeepInfer同事介绍文章

先打开index.html阅读，或用支持LaTeX的Markdown阅读器打开主文。正文9章40节、32组公式、13张PNG；第2章包含RMSNorm、HC/mHC、n-gram/Engram、投机解码与连续DFlash伪代码，标注输入输出和训练参数shape；附录A为D02入口26条加原四段72条，共98条kernel，附录B为71次主模型调用。

建议先沿正文理解输入到输出，再展开六处选读；第二遍做第8章三个trace核对。阅读版支持目录、折叠流程和Task跳转。HTML公式需联网加载MathJax；PNG和Markdown可离线读取。解压后保留assets和data目录。

正文和附录的shape按固定源码推导，耗时和事件来自2026-10-27采集。原始trace、日志及机器访问资料未打包。2026-11-04补全概念伪代码及DFlash启动与循环；固定trace结构与V4.1-Flash配置一致，正文已说明名称校正，没有新增NPU实验。

在线阅读：https://zhaoyuhsin.github.io/downloads/deepinfer-inference-guide-20261102.html
