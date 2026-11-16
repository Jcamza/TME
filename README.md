## 新增：MOS 四个 OOD 数据集的一键泛化实验（固定 α=0.7）

已新增 `scripts/05_check_mos_datasets.sh`（查找已有数据）、`scripts/06_prepare_mos_datasets.sh`（只创建官方路径的软链接视图，缺失数据可显式下载）、`scripts/07_run_all_mos.sh`（一次运行四个数据集）、`scripts/08_prepare_and_run_all_mos.sh`（可选一条命令完成）。

**使用原有 `iso` 环境，** 不改动 NegLabel 的核心源码、不做消融、不添加 Object-DINO。先运行：

```bash
cd /data/hx/code/NegLabel_Crop_Compare
conda activate iso
bash scripts/05_check_mos_datasets.sh
bash scripts/06_prepare_mos_datasets.sh --download-missing
bash scripts/07_run_all_mos.sh
```

如果确认所有数据已存在，去掉 `--download-missing`；如果希望所有步骤自动串联，执行 `bash scripts/08_prepare_and_run_all_mos.sh --download-missing`。**下载可能涉及较大数据文件**，只有显式添加 `--download-missing` 才会下载。阅读 [完整数据准备说明](docs/MOS_FOUR_DATASETS.md) 获取路径、MOS 官方来源、结果格式和注意事项。

---

# NegLabel-Crop-Compare (Official Baseline + Multi-Crop OOD Detection)

**目标：一条命令，在同一个模型、同一批图像、同一组负标签、同一组评分设置下，同时评估官方原始 NegLabel 与多视图裁剪融合 NegLabel，并自动输出 AUROC/FPR95 对照表。**

- 原始算法：Jiang et al., *Negative Label Guided OOD Detection with Pretrained Vision-Language Models*, **ICLR 2024**，官方 https://github.com/tmlr-group/NegLabel
- 裁剪方法来源：Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, **NeurIPS 2012**，Sec. 4.1 的多位置 Test-Time Cropping 思路；用 PyTorch `torchvision.transforms.FiveCrop` / `TenCrop` 实现。
- 用户附件的 Object-DINO：Rawlekar et al., *Finding Distributed Object-Centric Properties in Self-Supervised Transformers*, **CVPR 2026**。这篇论文用 Q/K/V 各层注意力头聚类生成物体定位图，并**不是 FiveCrop 的来源**。本项目首轮**未使用 Object-DINO 算法**，以避免额外模型引入不确定因素。具体说明见 `docs/METHOD_AND_SOURCES.md`。

> **交付范围 / 重要说明**：本 ZIP 包含 **完整的新增实验项目代码、Shell 脚本、评测、结果记录、绘图和单元测试**。为严格保持对照可靠性，本项目在运行时直接调用**官方原版** NegLabel 的模型、NegMining、Prompt、负标签和评分函数，**不复制或改写官方仓库的模型源文件**。因此运行真实实验必须另备官方 NegLabel checkout、其环境与 CLIP 权重，以及 ID/OOD 数据集。本 ZIP 不包含这些大型资源；也没有声称已经在 ImageNet 上跑出了性能提升。

## 1. 项目结构

```text
NegLabel_Crop_Compare/
├── README.md
├── config.env.example              # 服务器路径与实验参数
├── compare.py                      # Python 入口
├── requirements-extra.txt
├── scripts/
│   ├── _common.sh
│   ├── 00_check.sh                 # 环境和官方源码检查
│   ├── 01_compare.sh               # 一键原始 vs 裁剪比较
│   ├── 02_smoke.sh                 # 本地单元测试
│   ├── 03_download_official.sh     # 有网络时获取官方代码（不会覆盖）
│   └── 04_official_run.sh          # 原样执行官方 run.sh 的交叉核验
├── neglab_crop/
│   ├── preprocess.py              # 原图 Resize(224)+CenterCrop(224), FiveCrop/TenCrop
│   ├── data.py                    # 读取官方 ID/OOD 数据路径
│   ├── official.py                # 导入官方模型和 NegMining / grouping
│   ├── scoring.py                 # 图像特征融合、数学一致性测试函数
│   └── runner.py                  # 双分支检测+官方评价+CSV/PNG
├── tests/
│   ├── test_preprocess.py
│   ├── test_data.py
│   ├── test_scoring.py
│   └── test_runner_dummy.py
├── docs/METHOD_AND_SOURCES.md
└── results/
```

## 2. 环境要求

**直接使用原始 NegLabel 的 MMClassification + OpenAI CLIP 环境**（官方 README 建议 Python 3.8 / PyTorch 1.10 / CUDA 11.3；也可以自行验证现代环境）。与用户其他项目中的 `iso` 环境不一定兼容；尤其旧版 MMCV/MMClassification 与较新 PyTorch 的版本限制，请以官方环境先跑通为准。

GPU：CUDA。官方 `CLIPScalableClassifier` 在源码中指定 `cuda:LOCAL_RANK`，因此真实实验需要 GPU，CPU 环境仅用于单元测试。

先确认官方代码已经下载。若服务器不能联网，应先在有网络的机器上下载官方仓库压缩包并上传解压，例如解压到：

```bash
/data/hx/code/NegLabel
```

建议的目录：

```text
/data/hx/code/
├── NegLabel/                # 原始官方源码，不修改
│   ├── custom_config/clip.py
│   ├── mmcls/
│   ├── txtfiles/
│   └── data/                # 原始官方数据位置或软链接
└── NegLabel_Crop_Compare/  # 本项目源码
```

如果网络可用，也可以运行：

```bash
bash scripts/03_download_official.sh
```

它会从官方 GitHub 下载代码，**不会修改已存在的官方源文件**。

## 3. 配置（只需配置一次）

```bash
cd /data/hx/code/NegLabel_Crop_Compare
cp config.env.example config.env
nano config.env
```

关键参数：

```bash
OFFICIAL_ROOT=/data/hx/code/NegLabel
GPU=0
BATCH_SIZE=16
VIEW_BATCH_SIZE=32
WORKERS=2
ALPHA_GRID=0.2
FUSION=feature
CROP_MODE=fivecrop
OUTPUT_DIR=results/neglab_compare_fivecrop
```

如果官方 config 中的 `./data/val` 或 `./data/ood_data` 等路径与你的服务器不一致，建议**通过 `config.env` 覆盖数据路径**，不要修改官方源码：

```bash
ID_PATH=/data/Public/imagenet/val
ID_ANN=/data/Public/imagenet/meta/val_labeled.txt
OOD_INATURALIST=/data/Public/ood_data/iNaturalist/images
OOD_SUN=/data/Public/ood_data/SUN/images
OOD_PLACES=/data/Public/ood_data/Places/images
OOD_TEXTURES=/data/Public/ood_data/Textures/dtd/images_collate
```

这些仅是示例位置，需要替换为你已有的真实数据集路径。`ID_ANN` 格式是官方的 `<relative_image_path> <class_index>`，如 `ILSVRC2012_val_00000001.JPEG 65`。OOD 目录应直接包含相应数据集图片；不要把四个数据集混在同一个文件夹。

如果 CLIP 的 OpenAI 原生 `.pt` 权重没有缓存，可使用：

```bash
CLIP_WEIGHTS=/data/Public/models/clip/ViT-B-16.pt
```

**注意**：OpenAI CLIP `clip.load()` 接受原生 `.pt` 检查点。HuggingFace 的 `pytorch_model.bin` / `model.safetensors` 不能直接当作原生 `.pt` 传入。若只有 HF 版 CLIP 权重，需要另做严格等效的模型适配，此项目并未静默转换权重。

## 4. 一键运行

首先激活已安装好的原始 NegLabel 环境，例如：

```bash
conda activate open-mmlab
cd /data/hx/code/NegLabel_Crop_Compare
pip install -r requirements-extra.txt
bash scripts/00_check.sh
bash scripts/02_smoke.sh
```

实际 OOD 数据第一次试运行：

```bash
# 小规模 GPU+ImageNet+OOD 对照功能检查（此结果不能用于论文）
bash scripts/01_compare.sh --limit-id 32 --limit-ood 32 --dataset iNaturalist
```

完整 ImageNet-1k + 四个 OOD benchmark：

```bash
# 原始 NegLabel + FiveCrop 特征融合一次完成
bash scripts/01_compare.sh
```

或者手动运行：

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 \
python compare.py \
  --official-root /data/hx/code/NegLabel \
  --config custom_config/clip.py \
  --crop-mode fivecrop \
  --fusion feature \
  --alpha 0.2 \
  --batch-size 16 \
  --view-batch-size 32 \
  --workers 2 \
  --output-dir results/crop_compare \
  --check-official-parity
```

`--check-official-parity` 会在第一批样本上对比我们提取的原始特征评分与**官方 `ScalableClassifier.forward()`** 输出，并严格检查最大绝对误差。

若想研究融合权重敏感性，**不重复编码图片**：

```bash
bash scripts/01_compare.sh --alpha-grid 0,0.1,0.2,0.3,0.5
```

若要比较特征融合与分数融合：

```bash
bash scripts/01_compare.sh --fusion both --alpha-grid 0.2
```

`--crop-mode tencrop` 可将裁剪扩展为原始图 + 10 张裁剪图（成本约为 11 次图像编码）。

> 论文级实验应预先固定 `alpha`，不要使用测试集结果来选择最优超参数；权重网格仅用于消融/敏感性分析。运行时删除 `--limit-id`/`--limit-ood` 才表示全量评测。

## 5. 结果如何查看？

一次运行会输出：

```text
results/neglab_compare_fivecrop/
├── settings.json               # 具体配置、数据数量、负标签数量
├── comparison.csv              # 逐个 OOD 数据集 + AVERAGE, 包含 original 与 crop
├── comparison.json
├── auroc_comparison.png
├── fpr95_comparison.png
└── scores/
    ├── ID.csv                   # 每个 ID 图像原始/裁剪后的真实检测分数
    ├── iNaturalist.csv
    ├── SUN.csv
    ├── Places.csv
    └── Textures.csv
```

`comparison.csv` 字段：

- `dataset`, `method`, `AUROC`, `AUPR_IN`, `AUPR_OUT`, `FPR95`
- `n_id`, `n_ood`
- `delta_AUROC_vs_original`, `delta_FPR95_vs_original`

`official_original` 是当前官方代码分支跑出的真实基线，而不是论文表中直接复制的数值。

`feature_alpha_0.200` 是对同一批数据额外计算的裁剪融合分支。

最关键的判断是：**AUROC 越大越好，FPR95 越低越好。** `delta_AUROC_vs_original > 0`、`delta_FPR95_vs_original < 0` 是希望观察到的改善，但不保证出现。

## 6. 为什么对比是公平的？

1. **共享同一个原始模型实例**。原始和增强结果都来自它，不重新加载一份不同权重的 CLIP。
2. **共享负标签**。初始化阶段完全复用官方 NegMining 输出，绝不重新挖掘或更改文本提示词。
3. **原始分支的图像预处理完全保留**。与官方 `OODBaseDataset` ImageNet transform 一致。
4. **共享官方分组 Softmax 评分**。原始和融合特征最后都调用官方 `ScalableClassifier.grouping()`。
5. **共享官方评价代码**。导入原版 `mmcls.utils.ood_evaluate.evaluate_all`，包括其 FPR95 阈值选择策略。
6. **同一 DataLoader 逐图配对**，保留文件名及标签，可追查哪张图改善或退化。
7. **独立于原始仓库**。新的代码、输出都放在扩展项目，不会覆盖/篡改原始源码。

## 7. 已知边界

- 本地 CI/单元测试仅验证本项目的几何裁剪、特征融合、分组评分数值一致性、数据流与输出；本 ZIP 内**不包含**公开 ImageNet 和官方 CLIP 权重，因此未完成真实 GPU 基准验证。
- 官方 NegLabel 源码中 `neg_topk=0.15`、`pencentile=0.95` 等参数均取自官方 `custom_config/clip.py`，不要误认为我们的实现自动保证某个固定 10,000 负标签数量。`settings.json` 会保存最终实测数量。
- 原始官方某些数据路径、旧环境依赖以及无网络情况下 `clip.load()` 下载模型的问题仍需按官方 README 解决。
- FiveCrop 既可能凸显 ID 目标，也可能丢失定义 OOD 的上下文，因此改进并不必然。
- 用于速度对比时，必须把新增裁剪的 CLIP 前向计算计入总耗时，而不是仅报告 NegLabel 分组计算耗时。

完整理论机制与文献来源见 `docs/METHOD_AND_SOURCES.md`。

---

## New: CLIP CLS-70 and Object-DINO-guided region experiments

A separate runner (`neglab_crop.attention_runner`) adds **two training-free region proposal approaches** on top of the original NegLabel scorer, while preserving the previously verified FiveCrop result. **The validated `runner.py` is unchanged.**

See **[docs/ATTENTION_REGION_EXPERIMENT.md](docs/ATTENTION_REGION_EXPERIMENT.md)** for method definitions, source citations, limitations, model-weight preparation, exact commands, cache/resume mechanics, and inspection criteria.

Quick start from the existing `iso` environment after setting the pretrained DINOv2 model path in `config.env`:

```bash
bash scripts/09_check_attention_models.sh
bash scripts/10_attention_smoke.sh
bash scripts/11_attention_all_mos.sh
```
