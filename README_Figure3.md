# Figure 3 代码包 — Positive but Noisy / NegLabel–FiveCrop

本包用于 **已有逐样本 CSV 分数**，不提取视觉特征、不启动 CLIP、不跑 NegMining。请在项目根目录 `/data/hx/code/NegLabel_Crop_Compare` 中解压，沿用 conda `iso`。

## 目标、展示方式

主图采用 2 行 × 2 列：

| | Places | Textures |
|---|---|---|
| Original NegLabel | ImageNet ID vs Places OOD 的原始置信度密度 | ImageNet ID vs Textures OOD 的原始置信度密度 |
| FiveCrop + NegLabel | ID vs OOD 融合后密度 | ID vs OOD 融合后密度 |

- 蓝/橙（Matplotlib 默认前两色）区分 ID/OOD，横轴是 **ID confidence（越大越 ID）**。
- 每列上下两行使用完全相同的横轴范围、直方图 bin edges 与 y 轴范围，保证对比客观。
- 黑色虚线为「按官方 FPR95 对应、最接近 95% ID TPR 的阈值」，旁边显示真实 AUROC/FPR95；不是声称固定阈值在两个模型中相同。
- `raw` 使用原始评分作为横轴。`logit` 仅对绘图做单调变换 `log(S/(1-S))`，**计算指标始终使用原始 S**，避免分数极接近 0 或 1 时难以看清重叠区域。logit 绘图数值裁剪至 `[1e-10, 1-1e-10]` 仅为可视化数值稳定性。两种图都会输出，请根据视觉可读性选择一版投稿。
- ID 采用同一组 50,000 张 ImageNet val；**列间 ID 分布并非独立实验**。

Bootstrap：针对 Places 和 Textures **各自**以放回方式重采样 ID 及对应 OOD，Original 和 FiveCrop 共享相同图像抽样索引，默认 2,000 次，给出 ΔAUROC 与 ΔFPR95 的 2.5%–97.5% 百分位置信区间。只报告数据集层面的 CI，不把两数据集混成独立重复。

## 安装与先检查现有 CSV

```bash
conda activate iso
cd /data/hx/code/NegLabel_Crop_Compare
unzip -n NegLabel_Figure3_Patch.zip
chmod +x scripts/15_figure3_inspect.sh scripts/16_figure3_plot.sh
python -m unittest discover -s tests -p 'test_figure3*.py' -v

# 列出当前已保存的原始分数文件、列名、每个文件的第一行；不会推理
SCORES_DIR=results/mos_four_alpha07/scores \
  bash scripts/15_figure3_inspect.sh
```

**重点：当前交接文档没有提供原始 scores CSV 的实际列名和文件名。** 所以运行前必须先执行 `15_figure3_inspect.sh`，不要猜测 `original` / `fivecrop` 的实际字段，也不要拿 **Table 2 的 MCM CSV** 代替 NegLabel 分数。正式绘图应采用 `results/mos_four_alpha07/scores/` 下 Original 和 FiveCrop 的逐样本 **NegLabel 评分**。

## 自动识别支持的结构与运行

如果 `scores/` 中有 `ID.csv`、`Places.csv`、`Textures.csv` **等每集一个宽表**，且每表同时有 Original、FiveCrop 两个评分字段，且字段符合 `original`/`original_score`/`neglabel`、`fivecrop`/`fivecrop_score`/`ours` 等常见命名，脚本会自动识别（请以 inspect 的结果为准）：

```bash
FIGURE3_OUTPUT=results/figure3_analysis \
SCORES_DIR=results/mos_four_alpha07/scores \
BOOTSTRAP=2000 SEED=20261112 \
bash scripts/16_figure3_plot.sh 2>&1 | tee figure3_analysis.log
```

也可以选另一套现成分数：

```bash
SCORES_DIR=results/mos_four_attention_cls95_alpha07/scores \
bash scripts/15_figure3_inspect.sh
```

注意该目录可能包含 CLS/DINO 评分；**只能映射 Original 和 FiveCrop**，不能把 attention 分支当成 Ours。

## CSV 文件/列名不符合自动识别时：使用显式 JSON 映射

复制下面为项目根目录 `figure3_inputs.json`，把 `<...>` 替换为 inspect **实测的**文件路径和列名，JSON 示例不是对真实文件结构的断言：

```json
{
  "ID": {
    "file": "results/mos_four_alpha07/scores/<ID文件名>.csv",
    "original_col": "<Original评分列名>",
    "fivecrop_col": "<FiveCrop评分列名>",
    "key_col": "<图像路径列名>"
  },
  "Places": {
    "file": "results/mos_four_alpha07/scores/<Places文件名>.csv",
    "original_col": "<Original评分列名>",
    "fivecrop_col": "<FiveCrop评分列名>",
    "key_col": "<图像路径列名>"
  },
  "Textures": {
    "file": "results/mos_four_alpha07/scores/<Textures文件名>.csv",
    "original_col": "<Original评分列名>",
    "fivecrop_col": "<FiveCrop评分列名>",
    "key_col": "<图像路径列名>"
  }
}
```

如果每个数据集的 Original 与 FiveCrop 分在 **两个 CSV**，则将那个数据集改为：

```json
"Places": {
  "original": {
    "file": "results/.../places_original.csv",
    "column": "<原始评分列名>",
    "key_col": "<图像路径列名>"
  },
  "fivecrop": {
    "file": "results/.../places_fivecrop.csv",
    "column": "<融合评分列名>",
    "key_col": "<图像路径列名>"
  }
}
```

**两个独立 CSV 必须有可用于严格对应的图像唯一标识。** 脚本检查每个文件内 key 不重复、两个文件 key 集合完全一致，并按 key 对齐，不允许仅凭行序配对。宽表原始与融合评分已经在同一行，即使没有 key，也能进行逐行配对。

运行：

```bash
INPUT_CONFIG=figure3_inputs.json \
FIGURE3_OUTPUT=results/figure3_analysis \
BOOTSTRAP=2000 SEED=20261112 \
bash scripts/16_figure3_plot.sh 2>&1 | tee figure3_analysis.log
```

或直接使用 Python：

```bash
python -m figure3.run \
  --config figure3_inputs.json \
  --out results/figure3_analysis \
  --bootstrap 2000 --seed 20261112 \
  --neglabel-root /data/hx/code/NegLabel
```

## 输出

```
results/figure3_analysis/
├── figure3_distribution_raw.pdf    # 原始评分，矢量
├── figure3_distribution_raw.svg    # 可编辑矢量
├── figure3_distribution_raw.png    # 预览 220 dpi
├── figure3_distribution_logit.pdf  # 可选：单调 logit 变换后的展示
├── figure3_distribution_logit.svg
├── figure3_distribution_logit.png
├── figure3_metrics.csv             # 正式 AUROC/FPR95、95% ID 阈值
├── figure3_paired_bootstrap_ci.csv # 配对重采样 Δ 和 95% CI
├── figure3_summary.json           # 运行参数、样本数、指标
└── figure3_caption.txt            # 可改写到论文里的英文图注
```

输入完整数目必须是 `ID=50000`、`Places=10000`、`Textures=5640`，缺失时直接报错。默认还要求实算的 Original/FiveCrop 在 Places 与 Textures 的 AUROC/FPR95 不偏离交接文档的既有结果超过 0.20 个百分点；目的是防止错误列名/错误分数/数据集混用。确有经过验证的配置差异时可显式 `--no-reference-check`，但不能将差异当作原实验复现。**不要用 `--allow-subset` 的输出撰写论文结论**。

## 如需仅检查图表而暂不做 CI

```bash
BOOTSTRAP=0 bash scripts/16_figure3_plot.sh
```

## 所需环境

```
Python 3.9+
numpy
matplotlib
scikit-learn
```

`scikit-learn` 由官方 NegLabel 的 `evaluate_all` 需要；其实现从 `/data/hx/code/NegLabel/mmcls/utils/ood_evaluate.py` 动态读取，与已有主实验评价口径一致。绘图脚本在 CPU 运行，无 GPU 依赖；无需重新跑 CLIP。

## Figure 3 应如何解释

1. **图中看** OOD 分布与 ID 分布在判别阈值附近的重叠；绝不能仅凭 OOD 平均分数下降就推断 AUROC 一定改善。
2. **指标中看** Places/Textures 的 AUROC 增益、FPR95 降低是否和分数分布变化一致。
3. **统计中看** `figure3_paired_bootstrap_ci.csv` 的 ΔAUROC 与 ΔFPR95 95% CI 是否跨零；跨零就不能声称该指标的增益有统计确定性。
4. 图仅比较 Original vs FiveCrop **在同一个 NegLabel 评分函数下** 的行为；它不能单独证明“负标签自动消除视觉噪声”，也不是对 Table 2 的 MCM 组直接绘图。

**建议正文图名：** *Effect of Multi-View Evidence on ID–OOD Score Separation.*
