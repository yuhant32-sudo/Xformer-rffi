# 上游来源与修改记录

- 上游项目：https://github.com/gladzhang/Xformer
- 论文：Xformer: Hybrid X-Shaped Transformer for Image Denoising，ICLR 2024。
- 作者：Jiale Zhang, Yulun Zhang, Jinjin Gu, Jiahua Dong, Linghe Kong, Xiaokang Yang。
- 本次来源：用户工作区中的 `Xformer-main` 源码目录。目录未包含 Git 元数据，因此不声明对应某个上游 commit 或 tag。
- 本轮修改：复制源码并保留相对路径；原 README 原样保存在 `README_UPSTREAM.md`；新增研究 README、Git 忽略规则及本说明。未修改上游 Python、配置、图像和数据说明内容。
- 未包含用户论文 PDF、研究数据、训练权重或生成结果。

## 许可说明

上游 README 明确声明 Apache License 2.0，`setup.py` 亦使用该许可证标识。本地源码包没有独立 LICENSE 文件，保留原作者声明；许可证正文见 https://www.apache.org/licenses/LICENSE-2.0 。本说明不改变上游或其依赖组件的许可。

上游说明代码基于 Restormer 和 BasicSR，并要求同时遵守其许可证。已有文件内署名与声明保持不变：

- https://github.com/swz30/Restormer
- https://github.com/XPixelGroup/BasicSR

项目研究设想及实施计划见根目录 README；不得将上游图像降噪实验结果标为本项目的 RFFI 结果。
