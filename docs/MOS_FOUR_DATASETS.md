# NegLabel / MOS 四个 OOD 数据集：准备与一键泛化测试

**本扩展不做消融，也不使用 Object-DINO。** 固定原始 NegLabel vs FiveCrop feature fusion (alpha=0.7)。只需现有 `iso` 环境，不新建 Conda 环境。新脚本与源码以补丁形式添加到原 `NegLabel_Crop_Compare` 中。

## 1. 数据集、数量和来源

| Role | Dataset | Expected images | Where to find it |
| --- | --- | ---: | --- |
| ID | ImageNet-1k validation | 50,000 | `/data/Public/imagenet/val/` (confirmed by user's server log) |
| OOD | iNaturalist (MOS-selected) | 10,000 | [MOS repository](https://github.com/deeplearning-wisc/large_scale_ood) curated archive |
| OOD | SUN (MOS-selected) | 10,000 | MOS curated archive |
| OOD | Places (MOS-selected) | 10,000 | MOS curated archive |
| OOD | Textures (full DTD) | 5,640 | `/data/Public/DTD/images` (confirmed by user's server log) |

**重要**：SUN397、Places365、完整 iNaturalist 原始数据 **不是** MOS 使用的 10k OOD 子集，不可直接替代。图片数量正确是必要的，但不自动证明数据身份；新下载的 MOS 原始 tar 包提供可追溯来源。用户服务器的 iNaturalist、SUN、Places 是否已存在，必须运行服务器侧检查脚本才能确认；我们不能远程读取用户的服务器。

原始网址（MOS 官方 README）：

- `https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz`
- `https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz`
- `https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz`
- `https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz`

以上文件来自 MOS 数据集整理或 DTD 官方。部分网址可能因网络策略不可访问，准备脚本会报错而不是悄悄用原始全量数据替代。下载属于显式操作，检查不会联网。

## 2. 复制补丁到现有实验项目

如果使用补丁包，解压后将包内 `NegLabel_Crop_Compare/` 的内容合并到服务器同名目录。**不要直接将新补丁覆盖你的 `config.env`、历史 `results` 或现有 50k ImageNet 索引。** 补丁包不含上述文件。

```bash
conda activate iso
cd /data/hx/code/NegLabel_Crop_Compare
```

## 3. 先检查服务器已有 OOD 数据

```bash
bash scripts/05_check_mos_datasets.sh
```

列出 ImageNet 索引状态以及四个 OOD 数据集的实际路径、数量、检测状态；保存 `data/mos_dataset_status.json` 和 `.csv`。脚本默认查找 `/data/Public`、官方 OOD 路径和已下载的本地数据，最多深入四层文件夹，避免递归扫描 ImageNet 的全部文件。如果数据放在更深处，指定：

```bash
bash scripts/05_check_mos_datasets.sh --search-root /data/hx/code
```

如果知道已有数据路径，可以在下一步明确传入 `--source 'SUN=/data/other/MOS/SUN'`。只有在数目为 MOS 预期时才会认为数据可准备。

## 4. 准备数据：只创建软链接视图，不复制图片

**如果所有 OOD 图片已经存在：**

```bash
bash scripts/06_prepare_mos_datasets.sh
```

**如果 iNaturalist、SUN、Places 缺失，显式授权下载 MOS 官方压缩包：**

```bash
bash scripts/06_prepare_mos_datasets.sh --download-missing
```

可通过 `--source` 指定已有的具体路径以避免重复下载：

```bash
bash scripts/06_prepare_mos_datasets.sh \
  --source 'iNaturalist=/data/other/ood/iNaturalist' \
  --source 'SUN=/data/other/ood/SUN' \
  --source 'Places=/data/other/ood/Places'
```

`--download-missing` 使用 `curl`，支持断点续传；下载到 `data/mos_downloads/`，解压到 `data/mos_downloads/extracted/`。**该目录必须保留，因为后续通过软链接引用图片**。如果 HTTPS 不可访问，会再尝试官方 README 发布的 HTTP 链接（HTTP 传输不具备完整性保护）。脚本不删除已有数据集，不覆盖已有路径。下载前请检查磁盘空间。

脚本把所有图像转成零拷贝的**扁平软链接视图**，对应 NegLabel 官方配置：

```text
/data/hx/code/NegLabel/data/
├── ood_data/
│   ├── iNaturalist/images/              # 10000 image symlinks
│   ├── SUN/images/                      # 10000 image symlinks
│   ├── Places/images/                   # 10000 image symlinks
│   └── Textures/dtd/images_collate/     # 5640 image symlinks
```

ImageNet 使用原图路径 `/data/Public/imagenet/val` 和索引 `NegLabel_Crop_Compare/data/imagenet_val_ood_index.txt`。索引缺失时会自动生成；每张标注的 `-1` 是仅供 OOD 分数计算使用的占位符，**不可用于分类准确率计算**。

## 5. 一键完整四数据集评测（一次模型加载）

先执行准备，再运行：

```bash
bash scripts/07_run_all_mos.sh
```

它先严格校验完整数量，然后使用一次 `scripts/01_compare.sh` 调用，读取完整 50,000 张 ImageNet 和四个完整 OOD 数据集，不使用 `--limit-id` 或 `--limit-ood`。无权重搜索、无消融。原始 NegLabel 与 FiveCrop feature alpha=0.7 同批评估，保留官方一致性检查。

结果：

```text
results/mos_four_alpha07/
├── run.log
├── settings.json
├── comparison.csv          # 4 个 OOD + AVERAGE、两种方法
├── comparison.json
├── auroc_comparison.png
├── fpr95_comparison.png
└── scores/
    ├── ID.csv
    ├── iNaturalist.csv
    ├── SUN.csv
    ├── Places.csv
    └── Textures.csv
```

防止意外覆盖已有结果，若存在 `comparison.csv`，脚本拒绝覆盖；可设置新输出目录：

```bash
MOS_RUN_OUTPUT_DIR=results/mos_four_alpha07_repeat bash scripts/07_run_all_mos.sh
```

如果想**全部步骤连起来执行**（可能大规模下载！）：

```bash
bash scripts/08_prepare_and_run_all_mos.sh --download-missing
```

这条命令会先检查，然后下载缺失的 MOS 数据、准备软链接，再自动执行全部测试。如果不希望自动下载，去掉 `--download-missing`，缺失数据时会明确报错退出。

## 6. 评测注意事项

1. 固定 `alpha=0.7` 是根据之前 Textures 上探索得到的，因此 Textures 的性能不是未经调参的独立泛化验证。更重要的是在 iNaturalist/SUN/Places 上的表现。
2. 对同一 ID/OOD 样本配对比较 Original 与 FiveCrop，且使用官方模型、负标签、NegMining、评分与指标实现。之前用户日志首批官方 parity 为 `max_abs_error=0`，但不代表本扩展在服务器上已经真实运行。
3. 请勿将未经筛选的完整 SUN、Places、iNaturalist 用于发表层面的 MOS 结果。检查图片数量只是合理性检查。
4. 脚本不需要 root；若没有 `/data/hx/code/NegLabel/data/ood_data` 的写权限，联系管理员或调整目录访问权限，不建议自行更改其他用户目录权限。
5. 官方的 `vit_lt.py` 新版 timm 兼容性导入问题已经由用户在服务器局部修复，新增脚本不会修改这一行。

## 引用

- Jiang et al., “Negative Label Guided OOD Detection with Pretrained Vision-Language Models,” ICLR 2024, https://github.com/tmlr-group/NegLabel
- Huang and Li, “MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space,” CVPR 2021, https://github.com/deeplearning-wisc/large_scale_ood
- Cimpoi et al., “Describing Textures in the Wild,” CVPR 2014, https://www.robots.ox.ac.uk/~vgg/data/dtd/
