# CLS-Attention 95% / Object-DINO 融合权重实验（NegLabel）

## 目的与现有结论

已完成四个 MOS 数据集的 50,000 ID + 35,640 OOD 全量对比，得到
Original AUROC/FPR95 94.24/25.67，FiveCrop 94.60/24.17，
CLS95 mask 94.20/26.84，Object-DINO bbox 94.25/26.01（均为百分数）。
这些结果固定使用 `alpha=0.7`，只说明该权重下两种目标区域增强无稳定收益，不能推出所有 alpha 都无效。

这里扫描 alpha，并保证区域生成方式不变：CLS 仍为 `cls_mass=0.95, cls_view=mask`；DINO 仍为当前 kNN 图划分及 bbox 方法。融合公式：

`h_alpha = normalize((1-alpha)*h_global + alpha*h_region)`。

当 `alpha=0`，直接引用官方原始 NegLabel 分数，避免二次归一化产生数值漂移。

## 设计

采用 `alpha_grid = 0,0.05,0.1,0.2,0.3,0.5,0.7,1.0`，对 CLS 和 Object-DINO **同一张图像只提取一次**全局、CLS Mask、DINO BBox 特征，并在同批次计算所有 alpha 的 NegLabel 分数；不重复运行 CLIP/DINO 的区域发现过程。

旧的 `--alpha` 参数仍可单独运行，完全保持原有使用方法。若提供 `--alpha-grid`，仅对 CLS 和 DINO 分支扫描；若显式传入 `--include-fivecrop`，FiveCrop 仍固定在 `--alpha`（默认为 0.7），不额外搜索权重。默认扫描脚本不重复计算 FiveCrop，其已验证性能可直接引用 `results/mos_four_alpha07`。

## 安装

将压缩包 `NegLabel_CLS_DINO_AlphaSweep_Patch.zip` 放到 `/data/hx/code`，在该目录运行：

```bash
unzip -o NegLabel_CLS_DINO_AlphaSweep_Patch.zip
conda activate iso
cd /data/hx/code/NegLabel_Crop_Compare
python -m pytest -q tests/test_alpha_sweep.py tests/test_attention_runner_dummy.py
```

## 小样本功能验证（仅检验流程，不解读指标）

```bash
ATTENTION_ALPHA_OUTPUT=results/alpha_sweep_smoke \
DINO_MODEL=/data/Public/models/dinov2-base \
bash scripts/12_attention_alpha_sweep_mos.sh \
  --limit-id 16 --limit-ood 16
```

## 全量单次实验

```bash
ATTENTION_ALPHA_OUTPUT=results/alpha_sweep_cls95_dino \
DINO_MODEL=/data/Public/models/dinov2-base \
bash scripts/12_attention_alpha_sweep_mos.sh
```

脚本先检查四数据集，在一次模型/数据遍历内完成 α 扫描。两条分支各 8 个 alpha，加原始基线，共 17 个方法分数列。

输出：

- `comparison.csv`：官方 `evaluate_all` 计算的每数据集 AUROC/FPR95 等；
- `scores/ID.csv` 与 `scores/{iNaturalist,SUN,Places,Textures}.csv`：每张图像的配对分数，便于后续 Bootstrap 检查；
- `alpha_sweep_summary.csv`：所有数据集与平均值的两条 alpha 曲线数值；
- `alpha_curve_<dataset>_delta_AUROC.png`、`alpha_curve_<dataset>_delta_FPR95.png`：改善曲线；
- `run.log`、`settings.json`、`diagnostics/`：运行信息及区域统计。

可通过环境变量 `ATTENTION_ALPHA_GRID=0,0.05,0.1,0.2,0.3,0.5,0.7,1.0` 自定义权重。

## 结果解释及公平性

1. 先检查 `alpha=0` 对于 CLS/DINO 是否与 `official_original` 在**每张图像的分数**上完全一致，而不只看最终两位小数。
2. 在四数据集上观察连续权重的 AUROC 与 FPR95 趋势，识别较小 alpha 能否减少背景破坏；检查是否存在仅单数据集收益或 AUROC/FPR95 冲突。
3. **这是一组探索性测试集参数敏感性分析，不能把四个 OOD 测试集上的最优 alpha 当成无监督选择得到的正式泛化性能。** 如果要报告固定 alpha 的正式对比，应先定义不利用目标测试标签的参数选择规则（例如固定小权重或独立验证集），再统一评价所有 OOD 数据集。
4. 不能将 Object-DINO-guided bbox 与作者原始 Object-DINO 模块的结果直接等同。

## 兼容性与备份

新增独立的 `attention_alpha_runner.py` 以及 `12_attention_alpha_sweep_mos.sh`；不修改原有 `attention_runner.py`、`runner.py`、NegLabel 官方代码和模型/裁剪步骤。现有 `10_attention_smoke.sh` 和 `11_attention_all_mos.sh` 的运行方式保持不变。每次运行使用新的结果目录，避免覆盖以前的 full/alpha0.7 实验结果。
