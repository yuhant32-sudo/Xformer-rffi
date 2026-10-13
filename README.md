# Xformer-rffi

使用 Xformer 替换 SCUNet，研究低信噪比下的射频指纹识别（RFFI）。

**当前状态：研究起点。** 本仓库整理了本地 `Xformer-main` 中的上游图像恢复代码，并制定 RFFI 迁移计划。尚未实现 WiSig/WVD 数据管线、RFFI 微调脚本、ResNet18 分类训练或联合训练，也尚未产生 RFFI 实验结果。下文中的超参数是实验起点，不代表已验证的最优配置。

## 研究目标

参考论文 *Radio Frequency Fingerprint Identification in Low SNR Based on SCUNet* 的流程：

```text
原始 IQ → WVD 时频图 → SCUNet 降噪 → ResNet18 → 发射机身份
                         ↓ 替换
原始 IQ → WVD 时频图 → Xformer 降噪 → ResNet18 → 发射机身份
```

低 SNR 会淹没发射机硬件引入的细微差异。目标不仅是使时频图更干净，还要保留有助于区分设备的指纹，以提高独立测试集上的识别率。

SCUNet 本身已经结合 Swin Transformer 与卷积。本研究不是首次将 Transformer 引入该流程，而是检验 Xformer 的空间注意力、通道注意力双分支和双向连接单元（BCU），是否更适合 WVD 上的降噪与指纹保持。这里的“通道注意力”指网络特征通道，并非无线传播信道。

**研究假设：** 图像降噪预训练提供恢复先验；在 WVD 上微调能够适配信号域；进一步加入身份监督，可能减少去噪对指纹的损伤。以上均需通过消融实验验证，不能由图像 PSNR 提升直接推出识别率提升。

## 代码组织

保留上游代码的相对路径，便于复用原始脚本；本轮未修改网络实现。

```text
basicsr/                         上游网络、数据集、损失和训练框架
  models/archs/x_former_arch.py   上游灰度评估脚本实际导入的 Xformer
  models/archs/Xformer_arch.py    上游另一个架构文件，保留供核对
options/                         上游自然图像训练配置，非 RFFI 配置
evaluate/                        上游图像降噪评估脚本
datasets/                        上游数据准备脚本和说明
experiments/pretrained_models/   权重放置说明；权重不提交
figs/                            上游 README 使用的论文插图
README_UPSTREAM.md               原作者 README、安装说明和引用
UPSTREAM.md                      来源、修改范围与许可证说明
requirements.txt                上游依赖列表
setup.py / setup.cfg / VERSION   上游包配置
```

本地论文 PDF、WiSig 数据、模型权重、运行日志和生成结果不纳入本次提交。未来的 RFFI 原始数据与缓存建议放在被忽略的 `data/` 下。

## 预训练＋微调实施计划

### 1. 复现数据划分并构造配对数据

- 先按参考论文的受控条件，从 WiSig 选择同一天、同一个接收机下的 10 个发射机，每个 800 个长度为 256 的信号样本。按 8:1:1 划分时，扩增前为 6400/800/800；这不是多 SNR 扩增后的样本总量。
- **先划分原始数据，再切片/加噪扩增。** 同一原始包、重叠片段及其不同噪声版本必须属于同一划分，避免泄漏。保存 packet ID、设备、日期、接收机、SNR 和 split 的清单。
- 在 IQ 域加入 AWGN，再计算 WVD，初期使用 −6、−4、−2、0、2、4、6、8 dB。记录信号功率和噪声功率定义，注明是相对参考信号的注入 SNR，还是估计的最终 SNR；参考信号自身已有噪声。
- 用同一片段加噪前的高 SNR WVD 作为参考。它不是绝对无噪声真值；不要用理想调制信号替代而抹去硬件指纹。
- 保存数值时频矩阵而非带坐标、色条的图片。WVD 可能有负值和大动态范围，需要明确是否采用平滑、绝对值或有符号压缩，并验证其对指纹的影响。归一化统计只由训练集确定。
- 初期采用单通道 `[B, 1, H, W]`。信号长度 256 不等于已确定的图像尺寸。明确 WVD 网格、边界补齐与输出裁剪策略；时间轴和频率轴意义不同，不直接照搬自然图像的旋转/翻转增强。

### 2. 选择并验证预训练权重

从 [Xformer 官方仓库](https://github.com/gladzhang/Xformer) 的 [预训练模型目录](https://drive.google.com/drive/folders/1Ui_X1JoC5jxgE6k_hK7AZ2qrLo2I_OsD?usp=sharing) 获取灰度降噪权重，放入 `experiments/pretrained_models/`。记录下载来源、文件名和 SHA-256。

先以 `evaluate/test_gaussian_gray_denoising.py` 的显式导入与模型参数为参考，核对 checkpoint 中的参数键与张量形状，并严格加载。上游 `options/` 中部分配置与该测试脚本不同；架构目录还存在两个提供 Xformer 的文件，自动发现机制也需要核查。**不能只填写权重路径就假定训练配置与权重兼容。** 如需修改输入层，必须记录哪些参数无法复用，而非通过静默忽略错误加载。

图像噪声标准差 sigma=15/25/50 与 RF 信号 SNR（dB）不是同一个量，不能直接对应。

### 3. 第一阶段：WVD 去噪微调

- 输入含噪 WVD，目标为配对高 SNR 参考 WVD，以 L1 重建损失建立基线。
- 初期尝试全参数微调，Adam、学习率 `1e-4`、batch size `4` 作为参考起点；根据验证集稳定性和显存调整。冻结部分层可作为后续对照，不预设为必要条件。
- 实现专用 WVD 配对数据加载器；上游图像高斯加噪数据集不能直接代替 IQ 加噪后生成 WVD 的流程。适配数值范围、张量形状、padding 与裁剪。
- 在训练集混合多个 SNR，固定验证集噪声种子；只用验证集选择停止点及超参数。保存权重、配置、种子、数据清单和预处理版本。

参考论文明确说明微调 SCUNet 预训练模型，去噪阶段使用 Adam、`1e-4`、batch size `4`。但具体 checkpoint、冻结策略与 SCUNet 微调轮数未被完整披露；本项目不将这些未披露项视为已经复现。

### 4. 第二阶段：训练识别器

冻结完成微调的 Xformer，用其输出训练 ResNet18，采用设备身份交叉熵。所有去噪方法保持相同的数据划分、分类器初始化策略和训练预算，并记录 ResNet18 是否使用预训练权重。

先完成“去噪器微调 → 分类器训练”的分阶段基线，再评估联合优化。预训练的价值应以实验验证，不能假定自然图像先验一定有益。

### 5. 第三阶段：指纹保持优化

在基线基础上尝试：

```text
L = λrec · L1(恢复 WVD, 参考 WVD)
  + λcls · CE(分类器(恢复 WVD), 设备标签)
```

联合训练时分类梯度传入 Xformer，以身份判别目标约束恢复过程。进一步可加入冻结教师网络的特征一致性损失；教师只能使用训练集训练，损失权重只能使用验证集选择。

为区分“骨干优势”和“训练策略优势”，SCUNet 也应接受相同的指纹保持训练方案。分类损失改善并不自动证明恢复图保留了真实物理指纹，仍需独立条件测试。

## 实验与评价

| 对照 | 目的 |
| --- | --- |
| 原始 WVD + ResNet18 | 无去噪基线 |
| SCUNet 预训练＋微调 + ResNet18 | 复现参考路线 |
| 随机初始化 Xformer + WVD 训练 | 衡量预训练收益 |
| Xformer 图像预训练权重直接测试 | 衡量未经领域适配的迁移效果 |
| Xformer 预训练＋WVD 微调 | 核心替换方案 |
| SCUNet / Xformer + 相同身份约束 | 区分架构与目标函数的贡献 |

主要指标：各 SNR 下的设备识别准确率、macro-F1、混淆矩阵；至少 3 个随机种子，报告均值和标准差。各对照保持数据、分类器、预处理和预算一致。

辅助指标：WVD 重建 MAE/PSNR/SSIM（明确共同的数据范围及参考定义）、参数量、显存和相同硬件/输入尺寸下的推理时延。**图像指标提升不能替代识别指标。** 对含负值 WVD 的 SSIM/PSNR，必须先约定可复核的计算范围，不能逐图调整范围美化结果。

先验证同日同接收机结果，再做跨日期、跨接收机和噪声失配实验。同日数据扩增不能替代采集条件多样性；未知设备识别属于额外任务，应另行设计划分和指标。

## 环境与运行边界

上游 README 记录 Python 3.8 / PyTorch 1.8.0 及 CUDA 环境，保留了原安装命令，见 [README_UPSTREAM.md](README_UPSTREAM.md#installation)。建议在独立 Linux/CUDA 环境中复现并锁定实际依赖版本；当前依赖清单未完整锁版本，现代 Python/PyTorch 的兼容性尚未验证。

```bash
git clone https://github.com/yuhant32-sudo/Xformer-rffi.git
cd Xformer-rffi
# 然后按 README_UPSTREAM.md 创建独立环境并安装上游依赖。
```

现有 `basicsr/train.py` 和 `evaluate/` 面向自然图像任务。目前没有可直接运行的 RFFI 训练命令，不应将上游图像训练脚本当作已完成的 WVD 实验。

## 里程碑

- [x] 整理本地 Xformer 源码，保留上游说明与署名。
- [x] 明确预训练＋微调路线与公平实验设计。
- [ ] 确定 WiSig 文件、设备/日期/接收机和无泄漏划分。
- [ ] 实现 IQ 加噪、WVD 变换与配对数据加载。
- [ ] 核对灰度 checkpoint，并完成加载与形状检查。
- [ ] 复现 SCUNet 与无去噪基线。
- [ ] 完成 Xformer 微调、ResNet18 训练与预训练消融。
- [ ] 评估身份约束、跨条件泛化与计算成本。

## 来源与引用

- Jiale Zhang et al. **Xformer: Hybrid X-Shaped Transformer for Image Denoising**, ICLR 2024. [论文](https://arxiv.org/abs/2303.06440) · [官方代码](https://github.com/gladzhang/Xformer)
- **Radio Frequency Fingerprint Identification in Low SNR Based on SCUNet**, IEEE/CIC ICCC 2024：本研究参考的 WVD → SCUNet → ResNet18 流程及迁移微调设置；本文 PDF 未随仓库分发。
- **Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis**. [SCUNet 官方代码](https://github.com/cszn/SCUNet)
- [WiSig 数据集](https://cores.ee.ucla.edu/downloads/datasets/wisig/)

本仓库不是 Xformer 官方仓库。上游 README 声明 Apache 2.0，并要求遵守 Restormer 与 BasicSR 的相关许可；来源及本轮修改范围见 [UPSTREAM.md](UPSTREAM.md)。
