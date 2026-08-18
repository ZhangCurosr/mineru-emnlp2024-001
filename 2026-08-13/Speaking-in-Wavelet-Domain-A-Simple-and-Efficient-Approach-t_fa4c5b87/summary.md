---
title: "Speaking-in-Wavelet-Domain-A-Simple-and-Efficient-Approach-t"
source: https://aclanthology.org/2024.emnlp-main.9.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:30:47"
field: "语音扩散模型加速"
keywords: ["Diffusion Model", "Wavelet Transform", "Speech Synthesis", "Speech Enhancement", "Diffwave", "CDiffuSE", "Acceleration"]
innovations: ["将语音扩散模型的生成目标从时域迁移至小波域，零架构改动实现训练/推理约2倍加速", "设计低频增强器与多级小波加速器，分别在保性能前提下实现2倍与超5倍训练加速", "系统对比六种小波基在语音扩散任务中的表现，提供不同场景下的选型指南"]
benchmarks: ["LJSpeech", "VoiceBank-DEMAND", "VCTK"]
---

# 论文速读：Speaking-in-Wavelet-Domain-A-Simple-and-Efficient-Approach-t

## 一句话总结
论文提出将语音扩散模型的生成目标从原始时域信号迁移至**离散小波域（Wavelet Domain）**，在不修改模型架构的前提下，使 Diffwave 和 CDiffuSE 的训练与推理速度提升约 **2 倍**，并在语音合成与语音增强任务上达到可比或更优的性能；配合低频增强器与多级小波分解，最快可实现 **>5 倍**训练加速。

---

## 研究问题与动机
1. **DDPMs 在语音生成中训练/推理成本过高**：虽能产出高质量语音，但需数百至数千步去噪，阻碍实际部署与声音定制。
2. **现有加速方法偏重推理**：多数工作仅优化推理步数或修改模型结构，难以同时加速训练（对新增声音/场景微调至关重要）。
3. **图像领域的小波加速思路无法直接迁移**：图像特征尺寸小（64×64、256×256），而语音特征序列长，需从信号本身入手降维。
4. **缺乏对多类小波基的系统性探索**：不同小波的消失矩、平滑性与重建阶对语音扩散的影响尚未清晰。

---

## 核心贡献（创新点）
1. **Wavelet Diffwave / Wavelet CDiffuSE**：通过 DWT 将语音特征长度减半后送入原扩散模型，零架构改动即实现训练与推理速度约 **2× 加速**，性能相当或略优；与现有方法本质区别在于**从信号侧压缩而非修改网络**。
2. **低频语音增强器（Low-Frequency Enhancer / Frequency Bottleneck Block）**：在 DWT 之后、扩散之前对低频分量进行放大并抑制高频噪声，使多数小波（如 Haar、Coif1）在加速两倍的同时**超越原始模型性能**。
3. **多级小波加速器（Multi-Level Accelerator）**：对语音进行多级 DWT，将特征长度压缩至 **1/4**，训练速度提升 **>5 倍**，MOS 保持可接受水平；是本文在速度-质量权衡上的进一步探索。
4. **小波基的系统性分析与选型指南**：对比 Haar、bior1.1、bior1.3、coif1、db2、cdf53 六种小波，揭示了消失矩/平滑度与音色保真度、去噪能力之间的权衡关系，为不同应用场景提供明确推荐。

---

## 方法详解
### 3.1 小波变换与压缩
- 采用 **Cohen–Daubechies–Feauveau 5/3（cdf53）** 双正交小波（无损压缩常用）为例。
- 低通滤波器 $\bar{L} = [-\frac{1}{8}, \frac{2}{8}, \frac{6}{8}, \frac{2}{8}, -\frac{1}{8}]$，高通滤波器 $H = [\frac{1}{2}, 1, \frac{1}{2}]$。
- 对语音信号 $x \in \mathbb{R}^{1 \times 2x}$ 做 DWT：经滤波后下采样得到近似系数 $cA \in \mathbb{R}^{1 \times x}$ 与细节系数 $cD \in \mathbb{R}^{1 \times x}$，拼接为 $y \in \mathbb{R}^{2 \times x}$ 作为扩散目标。
- 推理结束后通过 **IWT** 将 $y_0$ 重构回原始波形 $x_0$。

### 3.2 小波域语音扩散方案
**语音合成（以 Diffwave 为例）**
- 前向扩散：$q(y_t | y_{t-1}) = \mathcal{N}(y_t; \sqrt{1-\beta_t}\, y_{t-1},\; \beta_t \mathbf{I})$。
- 反向扩散：$p_\theta(y_{t-1}|y_t) = \mathcal{N}(y_{t-1}; \mu_\theta(y_t, t),\; \sigma_\theta(y_t, t)^2 \mathbf{I})$，网络输出对应 $cA$ 和 $cD$ 两部分的均值与方差。
- 训练目标（简化 ELBO）：
  $$\min_\theta \mathbb{E}\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, y_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\; t\right)\right\|^2$$
  其中 $\epsilon$ 为 $2 \times X$ 矩阵，分别对应 $cA$ 和 $cD$ 的真噪声。

**语音增强（以 CDiffuSE 为例）**
- 对干净语音 $x_0$ 与含噪语音 $x_n$ 均做 DWT，得到 $y_0$ 与 $y_n$。
- 前向条件扩散：$q_{\text{diff}}(y_t|y_0, y_n) = \mathcal{N}\!\left(y_t;\; (1-m_t)\sqrt{\bar{\alpha}_t}\, y_0 + m_t\sqrt{\bar{\alpha}_t}\, y_n,\; \delta_t \mathbf{I}\right)$，$m_t$ 从 0 递增至 1。
- 反向采样：$\mu_\theta = c_{y_t} y_t + c_{y_n} y_n - c_{\epsilon_t} \epsilon_\theta(y_t, y_n, t)$，系数由 ELBO 推导（见附录 B）。

**低频增强器（§5.1）**
- Frequency Bottleneck Block 位于 DWT 之后，放大低频分量、压制高频成分，使噪声主导的高频部分被衰减。

**多级小波加速器（§5.2）**
- 两次 DWT 将特征长度压缩至 **1/4**、通道数增至 4；配套 Multi-level Low-Frequency Voice Enhancement Module（含多级残差块）进一步抑制高频噪声。

---

## 实验与结果
### 数据集
- **语音合成**：LJSpeech（13,100 条，约 24h），测试集 1,000 条，主观 MOS 30 人 × 20 示例。
- **语音增强**：VoiceBank-DEMAND（28 训练/2 测试说话人，SNR 0/5/10/15 dB），测试集 824 条；评估指标 **PESQ** 与 **DNSMos**。
- **多说话人泛化**：VCTK 数据集（额外验证）。

### 主要结果（Table 1，Base / Large）
| 任务 | 指标 | 原始模型 | Wavelet（最佳） | 提升/变化 |
|---|---|---|---|---|
| 语音增强（Base） | PESQ | 2.466 | DB2: 2.415 | −0.051 |
| 语音增强（Base） | DNSMos | 3.116 | Coif1: 3.125 | **+0.009** |
| 语音增强（Large） | DNSMos | 3.140 | Coif1: 3.196 | **+0.056** |
| 语音合成（Base） | MOS | 4.38±0.08 | Haar: 4.41±0.09 | **+0.03** |
| 语音合成（Large） | SN | 4.395 | Bior1.3: 4.403 | +0.008 |
| 训练时间（Base） | — | 481.8 s | ~248 s | **≈ 2× 加速** |
| 推理 RTF（Base） | — | 0.728 | ~0.402 | **≈ 2× 加速** |
| 训练时间（Large） | — | 997.7 s | ~507 s | **≈ 2× 加速** |
| 推理 RTF（Large） | — | 6.387 | ~3.366 | **≈ 2× 加速** |

### 多级加速（Table 2，Haar Base）
- **4C（两级分解，长度 1/4）**：训练时间 **65.35 s**（**5.1× 加速**），RTF 0.126，MOS 4.32±0.09（较原始略降）。

### 核心结论
- 所有小波基均带来约 **2× 速度提升**，且 GPU 显存占用减半。
- **Coif1** 在 DNSMos 上最佳，**DB2** 在 PESQ 上最佳；语音合成中 Haar / Bior1.3 表现突出。
- 低频增强器可进一步补偿性能损失，部分配置**超越原始模型**。

---

## 相关工作脉络
1. **Diffwave（Kong et al., 2020）**：本文语音合成实验的基础模型，Wavelet Diffwave 在其第一层卷积处做适配（保持 channel 数不变），其余结构完全保留。
2. **CDiffuSE（Lu et al., 2022）**：条件扩散语音增强基线，本文验证其在小波域的 ELBO 可等价推广（附录 B 给出系数推导）。
3. **FastDiff（Huang et al., 2022）**：专注推理加速的扩散 TTS；本文与之本质不同——**同时加速训练与推理**，且无需改动模型主干。
4. **Wavelet Score-based Generative Modeling（Guth et al., 2022）**：图像领域用小波加速 score-based 扩散；本文将其思想迁移至 **DDPM 原始音频波形生成**，并解决语音特征长、需兼顾音色保真等新挑战。
5. **Latent Diffusion Models（Rombach et al., 2022）**：通过 VAE 编码器压缩潜在空间；本文小波方法**无需额外训练编码器/解码器**，属轻量级信号侧替代方案。
6. **单步/少步扩散增强（Lay et al., 2023; Yen et al., 2023）**：侧重于减少采样步数；本文从信号表示入手，二者正交互补。

---

## 局限性与未来方向
1. **计时误差**：速度测试在大规模集群上进行，即便同为 V100 也存在硬件波动，绝对秒数可能略有偏差（论文承认此局限但不影响 2× 量级结论）。
2. **多级压缩导致音质下降**：两级 DWT（4C）虽实现 >5× 加速，但 MOS 较原始模型下降约 0.06，**不适用于对音质要求极高的场景**。
3. **泛化验证仍有限**：仅在 Diffwave 与 CDiffuSE 两个模型、两类任务上验证，其他语音扩散架构（如 DiffTTS、Speech Diffusion 大模型）的普适性待探索。
4. **小波基选择依赖任务**：论文指出不同小波在 PESQ / DNSMos / MOS 上表现各异，尚无统一最优解，需按具体需求（保真 vs 去噪）取舍。
5. **伦理影响**：降低语音合成门槛可能冲击广播、配音等行业就业（论文 Ethics Statement 已提及）。

---

## 研究启发与可借鉴点
1. **信号侧降维是加速生成模型的普适思路**：不修改网络、仅变换数据表示，可同时降显存、提并行效率；该思路可迁移至音频编解码、音乐生成、语音克隆等下游任务。
2. **低频增强器（Frequency Bottleneck Block）结构简单、易集成**：可作为通用预处理模块，与其他扩散 vocoder 结合；对"低频为主、高频多为噪声"的信号类型（语音、环境音）均有价值。
3. **小波基的系统性对比方法可复用**：本文对消失矩、平滑度、重建阶与性能关系的分析框架，可供后续研究者在选择小波时参考，避免盲目尝试。
4. **多级 DWT + 多级去噪模块的组合策略**：对需要极致推理速度的边缘设备部署场景（如移动端 TTS），两级/三级小波分解配合轻量增强器是一条可行路径。
5. **与小波包 /  learned wavelet 结合**：当前使用固定小波基，未来可探索自适应学习小波核，或与 LFCC、Mel 频谱等表征联合，进一步优化压缩-重建权衡。

---

## 关键术语表
**Discrete Wavelet Transform（DWT）**：离散小波变换，通过低通/高通滤波加下采样将信号分解为低频近似系数（cA）与高频细节系数（cD），实现时序压缩。

**Inverse Discrete Wavelet Transform（IWT）**：离散小波逆变换，从 cA 与 cD 经上采样与滤波重构原始信号，保证无损或极低失真重建。

**Diffwave**：Kong 等人提出的基于 DDPM 的通用语音合成声码器，直接在波形域进行扩散生成，是本文语音合成实验的基础模型。

**CDiffuSE**：Lu 等人提出的条件扩散语音增强模型，以含噪语音为条件去除噪声；本文在其小波域变体上验证方法通用性。

**Speech Naturalness（SN）**：基于 NISQA 深度学习模型的语音自然度预测指标，无需参考音频，可直接评估生成语音质量。

**PESQ（Perceptual Evaluation of Speech Quality）**：传统语音增强主观质量评价标准，需参考(clean)音频，取值越高音质越好。

**DNSMos**：ICASSP 2023 深度噪声抑制挑战赛提出的无参考语音质量评估指标，基于深度学习预测 MOS，对合成语音更友好。

**Real-Time Factor（RTF）**：推理时间与实际音频时长的比值，RTF < 1 表示推理快于实时；本文用于衡量采样速度。

---

## 可复现要素
- **数据集**：LJSpeech（公开）、VoiceBank-DEMAND（公开）、VCTK（公开）。
- **代码/权重**：**论文未提及**开源仓库与预训练权重。
- **关键超参（Diffwave Base）**：30 残差层、kernel size=3、dilation cycle=[1,2,…,512]、residual channels=64、diffusion steps=50、batch size=16、learning rate=2×10⁻⁴、Adam 优化器、训练 1M steps。
- **关键超参（Diffwave Large）**：residual channels=128、diffusion steps=200，其余同 Base。
- **关键超参（CDiffuSE Base）**：diffusion steps=50、batch size=16；**Large**：steps=200、batch size=15；训练 300K iterations，early stopping。
- **实验硬件**：32 × NVIDIA V100 32GB。
- **小波基列表**：Haar、bior1.1、bior1.3、coif1、db2、cdf53。

---
