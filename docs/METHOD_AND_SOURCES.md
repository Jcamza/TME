# Technical design and provenance

## 1. What is actually implemented

This is a **deterministic multi-view representation extension** of official NegLabel:

- Global reference view: `Resize(224) -> CenterCrop(224) -> ToTensor -> CLIP Normalize` (upstream `ood_dataset.py`).
- Crop views: `Resize(256) -> FiveCrop(224)` (four corners plus center), with optional `TenCrop(224)` including horizontal flips.
- Frozen CLIP encodes original and each crop; every embedding receives L2 normalization.
- No gradients, training, parameter optimization, new text labels, or new feature spaces.
- For feature fusion: `h = normalize((1-alpha)*h_global + alpha*mean(h_crops))`.
- Compute positive and negative similarities from **original official text embeddings** and feed them to the **original official `ScalableClassifier.grouping`**.
- Original scores and all fusion weights computed from exactly the same samples and model instance in one run.
- Alternative score fusion is a comparison baseline: score each crop with original NegLabel then interpolate scalar scores.

This is **not** Object-DINO; no Q/K/V head clustering, DINO object mask, or learned localization is claimed. The attached paper's theoretical motivation informs why foreground localization could be useful, but it is deferred until after deterministic multi-crop controls are tested. No results from Object-DINO are being used to assert gains for OOD detection.

## 2. Precise mathematical definition

`h0 = Norm(CLIP(x_original))`, `hr = Norm(CLIP(T_r(x)))`.

`h_alpha = Norm((1-alpha) h0 + alpha * mean_r hr)`.

`p_i=100 h_alpha^T e_i^+`, `n_j=100 h_alpha^T e_j^-`.

For each of the 100 groups of negative labels, use the **unmodified official grouping** `sum softmax(pos)`, then average the group scores.

Important: the official code uses temperature `t=1` on logits already multiplied by 100; it corresponds to `tau=0.01` on cosine similarities.

When `alpha=0`, the fused representation is the original normalized global representation, up to floating-point round-off.

## 3. Original sources and attributions

### Official baseline

- Jiang, Xue et al. **Negative Label Guided OOD Detection with Pretrained Vision-Language Models**, ICLR 2024. https://openreview.net/forum?id=xUO1HXz4an
- Official repository (Apache 2.0): https://github.com/tmlr-group/NegLabel
- Original global image preprocessing (the `Resize(224)/CenterCrop(224)` path): https://github.com/tmlr-group/NegLabel/blob/main/mmcls/datasets/ood_dataset.py
- CLIP embeddings, negative-label selection, official prompt behavior: https://github.com/tmlr-group/NegLabel/blob/main/mmcls/models/classifiers/multi_modal.py
- Unchanged OOD grouping function: https://github.com/tmlr-group/NegLabel/blob/main/mmcls/models/ood/clip_ood.py
- Default experiment settings (including `neg_topk=0.15`, `pencentile=0.95`, `ngroup=100`, `t=1`): https://github.com/tmlr-group/NegLabel/blob/main/custom_config/clip.py
- Original evaluation entry point: https://github.com/tmlr-group/NegLabel/blob/main/tools/ood_test.py
- Exact official evaluation metrics: https://github.com/tmlr-group/NegLabel/blob/main/mmcls/utils/ood_evaluate.py

### Cropping method

- Krizhevsky, Alex, Ilya Sutskever, Geoffrey Hinton. **ImageNet Classification with Deep Convolutional Neural Networks**, NeurIPS 2012, Sec. 4.1, test-time five spatial crops plus horizontal reflections. https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
- PyTorch official torchvision `FiveCrop`, `TenCrop` APIs: https://pytorch.org/vision/stable/generated/torchvision.transforms.FiveCrop.html

This project **uses the general deterministic multi-crop evaluation idea**, not AlexNet model architecture or the paper's original softmax-averaging rule. Our feature fusion is a new experimental extension, not a reproduction of an existing published algorithm.

### User-supplied supplementary reference

- Rawlekar, Samyak et al. **Finding Distributed Object-Centric Properties in Self-Supervised Transformers**, CVPR 2026 (Object-DINO). https://openaccess.thecvf.com/content/CVPR2026/html/Rawlekar_Finding_Distributed_Object-Centric_Properties_in_Self-Supervised_Transformers_CVPR_2026_paper.html
- Its key mechanism is **Q/K/V patch similarity and layer/head clustering**, not direct simple spatial cropping. That specific algorithm is **not** in this code, so please do not label this experiment `Object-DINO`.

## 4. Evaluation principles

All runs use exactly the same images, same known ID classes, same official selected negative embeddings, same temperature and the same seed-0 negative-label grouping. We only alter image-derived representation for the comparison method. Primary outputs: per-image original and modified OOD scores, AUROC, FPR95, and differences.

The production runner imports and uses the **exact official** `mmcls.utils.ood_evaluate.evaluate_all` for AUROC, AUPR_IN, AUPR_OUT, and FPR95. Unlike the common ROC threshold convention, the official FPR95 chooses the empirical recall level nearest to 0.95; this is intentionally preserved. Original scores are directly compared with official `ScalableClassifier.forward` via `--check-official-parity`.

## 5. Why no DINO in the first experiment?

Object-DINO is a training-free region proposal mechanism but requires its specific Q/K/V ensemble similarity, head-level k-means, object-cluster selection, and localization pipeline. Replacing it with final-layer attention or simple DINO foreground thresholding would NOT faithfully implement that paper. Fixed FiveCrop is auditable, low-dependency, and answers the primary hypothesis first. Object-DINO can later be added as a registered crop view generator if exact official code and a compatible local DINO checkpoint are available.
