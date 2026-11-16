# CLS-attention and Object-DINO-guided crops for NegLabel (2026-11-10)

## Scope and nonclaims

This module adds a controlled **training-free experiment** to the existing NegLabel_Crop_Compare project. It does **not** modify the original validated `neglab_crop/runner.py`, official NegLabel NegMining/labels/grouping, or the data preparation scripts. No weights are trained or fine-tuned; all crop proposals are image-only (no test labels).

**This Object-DINO branch is an ADAPTATION, not a bit-identical reproduction of the original Object-DINO discovery benchmark.** It follows the released Q/K/V multi-layer head selection implementation and then forms a single crop with a kNN normalized-cut graph. The authors' implementation is designed for unsupervised object discovery / MLLM grounding, not for NegLabel crop generation. Results must be named `Object-DINO-guided crop`, not `official Object-DINO reproduction`.

Sources:

- NegLabel: https://github.com/tmlr-group/NegLabel
- Object-DINO paper: Rawlekar et al., *Finding Distributed Object-Centric Properties in Self-Supervised Transformers*, CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Rawlekar_Finding_Distributed_Object-Centric_Properties_in_Self-Supervised_Transformers_CVPR_2026_paper.html
- Original source: https://github.com/SamyakR99/Object-DINO/blob/main/unsupervised_object_discovery/object_dino_feature_extraction.py
- OpenAI CLIP vision transformer: https://github.com/openai/CLIP/blob/main/clip/model.py
- DINOv2 backbone: https://huggingface.co/facebook/dinov2-base

## Data and setup

Use **the same four already-prepared MOS OOD data folders** from the current project, under the official-root `data/ood_data/...` symlinks. ID is the 50,000 ImageNet validation images at `/data/Public/imagenet/val` with `data/imagenet_val_ood_index.txt`; each OOD branch reads 10,000/10,000/10,000/5,640 images. Do not use SUN397 or generic Places365 instead of MOS subsets.

Conda environment is **iso** (no new environment). Required packages: `torch`, `torchvision`, `numpy`, `scikit-learn`, `scipy`, `Pillow`, and `transformers`. Existing `mmcv==1.7.2` and CLIP remain unchanged.

The DINOv2 backbone must be the **pretrained base** model with a HuggingFace `Dinov2Model` compatible configuration, e.g. `facebook/dinov2-base` (not a random initialized model). If it has already been cached, use it. To use a local model directory, set in `config.env`:

```
DINO_MODEL=/data/Public/models/dinov2-base
```

Local model folder requires `config.json` and `model.safetensors` or `pytorch_model.bin`. The runner defaults to `local_files_only=True`, so it never silently downloads large weights; for an explicit download use `--allow-model-download`. On a Windows machine, an alternate download procedure is:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/dinov2-base", local_dir="D:/dinov2-base")
```

Then upload that directory under `/data/Public/models/dinov2-base` and set `DINO_MODEL`.

## Experimental variants

1. `official_original`: official global `Resize(224)` and `CenterCrop(224)` feeding official CLIP ViT-B/16 with **exact original grouping**. Original parity is checked on first batch per dataset.
2. `fivecrop_alpha0.70`: existing FiveCrop pipeline (224 crop after resize shorter side 256), features mean pooled then fused 70% crop + 30% global. Included as previous reference.
3. `cls_attention_mask_mass70_alpha0.70`: obtain last-layer CLS→patch attention from CLIP's packed MHA Q,K projections. For CLIP-B/16, 14×14 patch grid. Remove the CLS self-attention and renormalize; select **minimum** top-ranked patches whose cumulative mass is at least 0.70. Replace unselected pixels with the CLIP RGB mean (zero in normalized input). Encode masked image with frozen CLIP. This is patch **retention/masking**, not rectangular cropping. Optional `--cls-view bbox` instead crops the enclosing rectangle and resizes to 224 (the rectangle may include unselected patches).
4. `object_dino_bbox_alpha0.70`: process exactly the same aligned 224×224 RGB image with frozen DINOv2-base (16×16 grid of 14×14 pixel patches). Capture Q,K,V projections from each layer and head. Compute mean-column Q/K/V self-similarity descriptors following the published author's code; standardize and cluster heads per image with KMeans (released code uses 12 clusters; experiment default `n_init=3` for runtime, set `--dino-n-init 10` to match its initialization setting). Select the cluster with the most final-layer heads. Aggregate selected Q/K/V patch affinity, form symmetric sparse kNN graph (`k=12`), compute normalized-cut Fiedler partition; select spatially compact, low-border foreground, use its largest connected component, construct padded bounding box, crop and resize to 224. If the graph is degenerate, use a deterministic saliency-based fallback and log it. This graph/box adaptation may behave differently from official TokenCut.

Every nonoriginal feature vector is independently normalized, and the fused image vector is

`h = L2norm(0.3 * h_global + 0.7 * h_region)`.

Original NegMining, WordNet, prompts, positive/negative text embeddings, official `ScalableClassifier.grouping`, and official `evaluate_all` metrics remain unchanged.

**It is not guaranteed that crop/masking improves OOD:** background and context can themselves be discriminative, especially in SUN/Places. Compare across all four datasets; do not tune the 70% CLS mass or alpha on the test data if claiming generalized test performance.

## First execution (one step at a time)

```bash
conda activate iso
cd /data/hx/code/NegLabel_Crop_Compare
# Copy the new files from the patch ZIP to your project, preserving old files.
python -m pytest -q tests/test_attention_regions.py
bash scripts/09_check_attention_models.sh
```

Then run the *small paired GPU smoke* over 16 ID + 16 from each OOD dataset:

```bash
bash scripts/10_attention_smoke.sh
```

If the DINOv2 model is unavailable, `--skip-dino` can verify CLIP/CLS/FiveCrop first:

```bash
bash scripts/10_attention_smoke.sh --skip-dino
```

**Always distinguish partial smoke metrics from full benchmarks.** After DINOv2 weights load and the full smoke is successful, run the complete benchmark:

```bash
bash scripts/11_attention_all_mos.sh
```

`ATTENTION_OUTPUT` environment variable selects a fresh results directory; `ATTENTION_SMOKE_OUTPUT` does the same for smoke. To compare CLS bbox instead of the strict mask, pass `--cls-view bbox` with a fresh `--output-dir`.

## Output

```
results/mos_four_attention_alpha07/
    settings.json
    comparison.csv
    comparison.json
    run.log
    auroc_comparison.png
    fpr95_comparison.png
    scores/ID.csv
    scores/iNaturalist.csv
    scores/SUN.csv
    scores/Places.csv
    scores/Textures.csv
    diagnostics/ID.csv
    diagnostics/iNaturalist.csv
    diagnostics/SUN.csv
    diagnostics/Places.csv
    diagnostics/Textures.csv
```

The `diagnostics` file records number and fraction of retained CLS patches, DINO bbox coordinates and area, count of selected object heads, and fallback cause. Inspect `dino_fallback` frequencies and `dino_box_fraction` before interpreting improvements. Any unexpectedly high fraction of full-image boxes means the object localization may be degenerating.

The default SQLite region proposal cache `data/attention_region_cache.sqlite` allows resuming proposal extraction across repeated runs; key includes method, image path, mtime/size and DINO settings. Do **not** delete the cache before restarting or benchmarking another OOD set. Delete it if the DINO weights change while keeping the same model path.

The Object-DINO stage is more computationally expensive than FiveCrop, since it performs Q/K/V extraction, per-image KMeans, and an eigendecomposition. Test with `--limit-id` and `--limit-ood` before the full run. CUDA RTX 4090 can run the two frozen backbones in a single process, but GPU memory / throughput depend on your setup.

## Acceptance conditions

- `Official parity` reports max_abs_error <= 1e-4 on first batch of each dataset.
- Same 50k ID, 10k iNaturalist, 10k SUN, 10k Places, 5640 Textures.
- No random backbone initialization. No label access for selecting the regions.
- `comparison.csv` contains all four planned methods when `--include-fivecrop` is used (original, CLS70, Object-DINO-guided, FiveCrop).
- `diagnostics/*.csv` has all image rows with region statistics; inspect fallback rate.
- Treat results as exploratory until full end-to-end run on the user's 4090, which has **not** been performed in the file-building environment.

For rapid sanity-checks, inspect the generated `examples/<dataset>_0000.jpg` collages: the original aligned crop, CLIP CLS 70%-mass retained pixels, and a red Object-DINO bounding box over the original image. The runner also reports `diagnostics/<dataset>_summary.json` (mean CLS retention, DINO fallback rate and mean cropped area). A method that selects near-full images or falls back frequently should not be interpreted as discovering accurate objects.

If the `transformers` import is missing, install `python -m pip install -r requirements-attention.txt` **inside the existing iso environment**, without changing Torch/MMCV/CLIP. The local model must be a **pretrained** DINOv2 checkpoint, not only a model configuration.

The α=0.7 fusion setting was selected during the previous Textures exploration, so Textures is not an untouched hyperparameter-selection test set. Cross-dataset improvements should be reported with that limitation.
