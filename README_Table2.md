# Table 2 — ID-only / FiveCrop 对照实验补充包

**目的：** 只新跑两组：`ID-only MCM` 和 `ID-only MCM + FiveCrop`。两组在每张图上共享同一次全局和五裁剪特征提取；旧实验 `NegLabel` / `Ours` 不再跑。最终输出四数据集 AUROC、FPR95 和正式 LaTeX Table 2。

## 1. 拷贝（覆盖：无）

整个压缩包按项目根目录结构打包；直接解压到 `/data/hx/code/NegLabel_Crop_Compare`。只新增 `table2_idonly/`、`scripts/13_table2_idonly_all_mos.sh`、`scripts/14_table2_merge.sh`、`tests/test_table2.py`、`README_Table2.md`，**不修改** `neglab_crop`、官方 NegLabel、已有配置或结果。

```bash
conda activate iso
cd /data/hx/code/NegLabel_Crop_Compare
unzip -n /path/to/NegLabel_Table2_IDOnly_Patch.zip
chmod +x scripts/13_table2_idonly_all_mos.sh scripts/14_table2_merge.sh
python -m unittest discover -s tests -p 'test_table2*.py' -v
```

依赖：当前 `iso` 已运行 NegLabel，含 `torch`、`torchvision`、`PIL`、`numpy`、`scikit-learn`、`tqdm`、OpenAI `clip`。**不新建环境、不要求 DINO**。要用官方 OpenAI CLIP 的 `ViT-B/16` 权重及缓存，不能直接把 HuggingFace 的 `clip-vit-base-patch16` 目录当作 OpenAI CLIP `.pt` 权重。

## 2. 三个阶段

### (a) 预检查（无模型推理）

```bash
python -m table2_idonly.run --dataset all --check-only
```

预期：ID=50,000；iNaturalist=10,000；SUN=10,000；Places=10,000；Textures=5,640；正文本类名取自官方 `NegLabel/mmcls/models/classifiers/class_names.py`，`prompt_idx=85`。如果服务器目录与交接文档不同，可通过 `--root-id`、`--root-inaturalist`、`--root-sun`、`--root-places`、`--root-textures` 指定。脚本会在数量不符时中断，避免错误数据源被当作正式实验。

### (b) 全数据集 smoke（每数据集仅 16 张；**不是正式成绩**）

```bash
CUDA_VISIBLE_DEVICES=0 TABLE2_OUTPUT=results/table2_idonly_smoke \
  MAX_SAMPLES=16 BATCH_SIZE=8 NUM_WORKERS=2 \
  bash scripts/13_table2_idonly_all_mos.sh
```

若使用 GPU 3，则将 `CUDA_VISIBLE_DEVICES=0` 改成 `CUDA_VISIBLE_DEVICES=3`，脚本内部仍用 `cuda:0`（CUDA_VISIBLE_DEVICES 重新映射）。smoke 不生成正式 Table 2 表。

### (c) 正式全部 MOS（一次完成 E1/E2）

```bash
CUDA_VISIBLE_DEVICES=0 TABLE2_OUTPUT=results/table2_idonly_alpha07 \
  BATCH_SIZE=16 NUM_WORKERS=4 \
  bash scripts/13_table2_idonly_all_mos.sh \
  2>&1 | tee table2_idonly_alpha07.log
```

启动后依次处理 ID 与四个 OOD；在每张图像上读取 1 个全局视图和 5 个局部视图，以同一冻结 CLIP 完成编码。`BATCH_SIZE` 默认 16，4090 上可按显存逐步调整为 24/32，OOM 就降到 8。**不会重复编码负文本，也不会执行官方分组 Softmax**。已经完成的 CSV 会校验 manifest 并自动跳过；中断时可以直接重复相同命令从批次边界续跑。

若服务器 CLIP 权重不是默认缓存位置：

```bash
CLIP_MODEL=/actual/path/to/ViT-B-16.pt \
  bash scripts/13_table2_idonly_all_mos.sh
```

需确保对应 checkpoint 与已有 NegLabel 实验 **完全同源**。

## 3. 数学定义（与交接文档一致）

- 官方 ImageNet-1k 类名和 `prompt_templates[85]`，逐类别调用 `clip.tokenize` / `encode_text`，并 L2 normalize → `U` `[1000,512]`（仅 CLIP ViT-B/16）。无需 WordNet，也无需 NegMining。
- 全局：RGB, `Resize(short_side=224, BICUBIC)` + `CenterCrop(224)` + CLIP Normalize。
- FiveCrop：RGB, `Resize(short_side=256, BICUBIC)` + `FiveCrop(224)` + CLIP Normalize；五个位置按 torchvision 的 TL / TR / BL / BR / center 顺序，**无镜像/随机增强**。
- 所有单视图图像特征先 L2 normalize 后求均值；融合 `h=.normalize(0.3*h0+0.7*mean(h1..h5))`；局部均值不单独归一化。
- 两个分数统一为 `MCM(h)=max(softmax(100 * h @ U.T))`，**高分 = ID**。E1 用 `h0`，E2 用 `h0.7`。
- 评价直接动态载入官方 `NegLabel/mmcls/utils/ood_evaluate.py` 中 `evaluate_all`，计算 AUROC / AUPR-IN / AUPR-OUT / FPR95；四 OOD **逐集单独计算，宏平均**，FPR95 保持官方离散阈值。

**精确对齐声明：** 交接文档没有附带 `neglab_crop/preprocess.py` / `runner.py` 的实际源码。本包依据交接文档写出相同算子，并明确采用 OpenAI CLIP 常用 BICUBIC 插值和 float32 归一化；该两项还应在服务器对照当前版本源代码核验。若你原代码采用不同的插值或图像特征 dtype 归一化顺序，应以实际 E3/E4 的实现为准，同步改这里，**不要把预处理变更当成 Table 2 的性能差异**。

## 4. 输出文件

`results/table2_idonly_alpha07/`：

| 文件 | 含义 |
|---|---|
| `ID.csv` | 50,000 张 ID 的文件相对路径 + 两组 MCM 得分 |
| `{iNaturalist,SUN,Places,Textures}.csv` | 各 OOD 文件路径 + 两组 MCM 得分 |
| `*.meta.json` | 各数据集 image manifest SHA256、prompt SHA256、预处理、模型、alpha、是否完成 |
| `table2_long.csv` | 4 方法 ×（4 OOD + Mean）：AUROC/FPR95，E1/E2 附 AUPR |
| `table2_paper.tex` | 可复制到论文的 4 行 LaTeX Table 2 |
| `table2_deltas.csv` | 四种成对比较，delta AUROC / delta FPR95（百分点） |
| `table2_report.txt` | 简洁结果和指标差值 |

**保留逐样本分数**，后续能用于 Fig. 3 或附加分析；每张原始图只会在一个数据集上计算一次。

## 5. 已有 E3/E4 怎么合并？

正式运行结束自动调用 `scripts/14_table2_merge.sh`。它通过附件交接文档的**真实已有、但仅保留两位小数的历史结果**（见 `table2_idonly/reference_metrics_from_handoff.csv`）填充 E3/E4，不重新推理。若已从 `results/mos_four_alpha07/` 导出**原始完整精度** `dataset,method,auroc,fpr95,source` CSV，请将 `REFERENCE_CSV` 指向该文件并仅重建表：

```bash
REFERENCE_CSV=/path/to/reference_metrics_precise.csv \
  bash scripts/14_table2_merge.sh
```

参考文件应包含 10 行：方法 `NegLabel`、`Ours` × 数据集 `iNaturalist`、`SUN`、`Places`、`Textures`、`Average`。`Average` 采用原日志未舍入宏平均，避免四舍五入导致的 0.01 pp 偏差。该合并**不会把过去的数字虚构为本次测得**；`source` 字段保留出处。只为生成 Table 2，不改变原论文 Table 1。

## 6. 论文应如何解释

优先根据四组实际数据判断：MCM 本身是否受益于 FiveCrop（E2-E1）；NegLabel 是否受益于 FiveCrop（E4-E3）；负文本在 global / fivecrop 上是否有效（E3-E1 与 E4-E2）。**不要**把 `(E4-E3)-(E2-E1)` 当作已经证实的严格因果互补性，也不要声称 E1/E2 是“NegLabel 去掉负文本后直接运行分组评分”：它们是标准 MCM 检测器，评分形式不同。若某组结果不满足预期，应如实报告。

## 7. 故障排除

- `ModuleNotFoundError: clip`：核对 `iso` 环境是否为旧实验真实运行环境；不要新建 conda。
- `CLIP model not found` / 尝试下载：用 `CLIP_MODEL=/path/to/correct/OpenAI-CLIP-checkpoint.pt`，或者确认原先运行时已缓存在 `~/.cache/clip/`。
- image count mismatches：先用现有 `bash scripts/05_check_mos_datasets.sh` 检查，必要时传 `--root-X`；**不要**默默忽略计数。
- 结果已经存在但 alpha/model/config 改了：选择新 `TABLE2_OUTPUT` 目录，或者显示传 `--no-resume` 全部重算。
- `CUDA out of memory`：设 `BATCH_SIZE=8` 或 `4`。
- `report.py` 发现数据不全：先完成五个数据集全部推理，不应从 smoke 表填充论文。
