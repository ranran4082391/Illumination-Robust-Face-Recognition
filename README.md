# YinYangFace: Qwen-0.5B Perceptor Guided Relighting for Half-Face Illumination

> Illumination-Robust Face Recognition under Half-Face ("Yin-Yang") Lighting,
> via a Frozen Qwen2.5-0.5B Perceptor + FiLM Fusion on top of ArcFace embeddings,
> with SH-spherical-harmonics physical relighting.

## What is this

A proof-of-concept demo tackling the failure mode of face recognition under half-lit
("yin-yang") lighting — where one side of the face is bright and the other falls
into shadow. The pipeline:

- **Face Encoder**: frozen InsightFace ArcFace `w600k_r50` (ONNX, 512-d embedding)
- **Perceptor**: frozen `Qwen/Qwen2.5-0.5B` (896-d hidden), fed a natural-language
  prompt describing the lighting direction; the last-token hidden state is used
  as a lighting condition vector
- **Fusion**: FiLM module — γ/β generated from the 128-d lighting condition
  modulate the 512-d ArcFace embedding: `h = e · (1 + γ) + β`
- **Relighting**: spherical-harmonics (SH, L=2, 9 coefficients) physically-based
  relighting over an ellipsoidal normal template, replacing naive hard-mask
  brightness cuts
- **Four-method comparison**: Frozen ArcFace / Global FiLM / CLAHE+FiLM /
  Ours (Qwen Perceptor + FiLM), evaluated under Easy and Hard gallery protocols

## Key Results (measured, not cherry-picked)

### Easy Protocol (gallery = all orig images per identity, max-pooled, 31 images)

| Method | Top-1 | EER | pos_sim | neg_sim | Trainable params |
|---|---|---|---|---|---|
| A. Frozen ArcFace | 1.0000 | 0.0011 | 0.8213 | 0.2420 | 0 |
| B. ArcFace + Global FiLM | 1.0000 | 0.0000 | 0.8183 | 0.2277 | 13,336 |
| C. CLAHE + ArcFace + FiLM | 1.0000 | 0.0000 | 0.8027 | 0.2239 | 13,336 |
| D. **Ours (Qwen Perceptor + FiLM)** | 1.0000 | 0.0000 | 0.8108 | 0.2236 | 292,248 |

→ All four methods saturate to 1.0 under the strong-gallery setting; no
discriminative difference.

### Hard Protocol (gallery = one orig image per identity, 6 images)

| Method | Top-1 | EER | pos_sim | neg_sim |
|---|---|---|---|---|
| A. Frozen ArcFace | 0.7957 | 0.1419 | 0.458 | 0.177 |
| B. ArcFace + Global FiLM | 0.8172 | 0.1484 | 0.451 | 0.168 |
| C. CLAHE + ArcFace + FiLM | **0.8387** | 0.1699 | 0.443 | 0.160 |
| D. **Ours (Qwen Perceptor + FiLM)** | 0.7527 | **0.1312** | 0.429 | **0.156** |

### Honest findings (counter-intuitive)

- **Top-1 ranking under Hard protocol: C (0.839) > B (0.817) > A (0.796) > D (0.753)** —
  Ours has the *lowest* top-1.
- **EER ranking under Hard protocol: D (0.131) < A (0.142) < B (0.148) < C (0.170)** —
  Ours has the *lowest* EER.
- **Why**: the FiLM fusion in D compresses the whole embedding space together
  (pos_sim drops 0.458→0.429, neg_sim drops 0.177→0.156). Threshold-based
  classification margin improves (best EER), but 6-way nearest-neighbor retrieval
  suffers because same-class and different-class embeddings are pulled together.
  This stems from training D with a 24-class classification head rather than a
  metric-learning objective — the fused h-space is not friendly to unseen-identity
  nearest-neighbor geometry.
- Qwen conditioning did learn lighting direction (D's CE loss converges to ~0
  while B/C stall at ~2.0), but that extra discriminative signal did not
  translate into a top-1 retrieval advantage.

## Directory Structure

```
.
├── index.html                  # Single-file web demo (browser-open directly)
├── index_v1.html               # Previous demo version (backup)
├── report.pdf                  # Chinese technical report (14 pages)
├── report_en.pdf               # English technical report (12-15 pages)
├── README.md                   # Chinese README
├── README_en.md                # This file
├── assets/
│   ├── metrics.json            # Headline metrics for Ours (Method D)
│   ├── methods_compare.json   # 4 methods × 2 protocols (easy + hard)
│   └── embedding_pca.json      # PCA projection of Method-D fused embeddings
├── data/
│   ├── meta.json               # 480 records (relit_path / prompt / light_dir)
│   ├── identities.json
│   ├── raw/                    # Original LFW aligned images
│   ├── train/                  # 24 identities (orig / _yy hard-mask / _relit SH)
│   └── val/                    # 6 unseen identities
├── figures/
│   ├── yinyang_examples.png    # Hard-mask examples
│   ├── relight_vs_mask.png     # Original | Hard Mask | SH Relit 3-panel
│   ├── train_curve.png         # B/C/D training loss curves
│   ├── embedding_pca.png       # PCA of Method-D fused embeddings
│   ├── methods_compare.png     # 4 methods × 2 protocols 2×2 bar chart
│   └── qualitative_cases.png   # 4 queries × 4 methods predictions
├── src/
│   ├── data_prep.py            # Download LFW + generate hard-mask yin-yang
│   ├── relight.py              # SH spherical-harmonics relighting core
│   ├── generate_relit.py       # Batch-generate SH relit images
│   ├── model.py                # FiLM / QwenPerceptor model definitions
│   ├── prep_emb.py             # Pre-compute ArcFace + Qwen embedding caches
│   ├── train.py                # End-to-end training of Method D
│   ├── compare_methods.py      # Easy-protocol 4-method comparison
│   ├── compare_hard.py          # Hard-protocol 4-method comparison
│   ├── eval.py                 # Re-evaluate metrics from caches
│   ├── infer.py                # Single image + prompt → top-k identities
│   ├── make_figs.py            # Training curves / PCA / qualitative cases
│   └── make_relight_fig.py     # Relight vs hard-mask comparison figure
└── weights/
    ├── arcface_dl/w600k_r50.onnx       # Frozen ArcFace (174 MB)
    ├── film_global_B.pt                # Method B weights
    ├── film_clahe_C.pt                 # Method C weights
    ├── film_fusion_D.pt                # Method D weights
    ├── face_emb_relit_cache.npy        # 480×512 relit ArcFace embeddings
    ├── face_emb_relit_clahe_cache.npy  # CLAHE-preprocessed embeddings
    ├── orig_emb_cache.npy              # 160×512 original embeddings
    └── qwen_prompt_cache_relit.pt      # SH-prompt Qwen embeddings
```

## Requirements

- Python 3.12, CPU-only (4 cores, 8 GB RAM)
- Packages: `torch torchvision transformers onnxruntime opencv-python
  scikit-learn matplotlib pillow numpy`

## Reproduction

```bash
# 1. Download LFW subset and generate hard-mask yin-yang images
python3 src/data_prep.py

# 2. Generate SH physically-relit images (480 samples)
python3 src/generate_relit.py

# 3. Pre-compute ArcFace + Qwen embedding caches
python3 src/prep_emb.py

# 4. Train Method D and run Easy-protocol 4-method comparison
python3 src/compare_methods.py

# 5. Run Hard protocol (6 gallery, one per identity)
python3 src/compare_hard.py

# 6. Render all figures
python3 src/make_figs.py
python3 src/make_relight_fig.py

# 7. Launch the web demo
python3 -m http.server 8000
# Open http://localhost:8000/index.html in a browser
```

## SH Spherical-Harmonics Relighting

The previous version used a naive left/right mask multiplied by brightness,
which is physically unscientific. The new version:

1. **Ellipsoidal normal template**: for a 112×112 aligned front face, construct
   a fixed ellipsoidal surface normal map (center protruding, edges receding) as
   an approximation of facial surface normals `n`.
2. **SH lighting L=2**: 9 basis functions Y_k, with randomly sampled coefficients c_k.
3. **Lambertian rendering**: `I = Σ_k c_k · Y_k(n)`, output = original albedo
   × max(0, I) + ambient floor.
4. **Randomization**: main light direction sampled on a hemisphere
   (large x-component → side light; large y-component → top/bottom light),
   intensity [0.8, 1.5], ambient [0.1, 0.3], p85 exposure normalization.

This produces continuous shadow gradients across the nose bridge, cheekbones,
and eye sockets, rather than a hard half-image brightness cut.

## Web Demo

Open `index.html` directly in a browser (no backend required). The page
supports:

- Drag-and-drop image upload
- 4 preset lighting-direction samples (upper-left / right / top / bottom)
- Side-by-side original vs. relit (yin-yang) visualization
- Top-3 recognition results with confidence bars
- 2D PCA scatter plot with the current query highlighted
- Easy / Hard protocol toggle for the 4-method comparison bar chart

## Limitations & Future Work

- Qwen is a frozen text encoder; prompts are auto-generated from the SH light
  direction. The model never actually sees the image — this is a proof-of-concept.
- Under the Hard protocol, Method D's top-1 is actually the lowest: classification-head
  training is not suited to open-set retrieval. Future work should switch to
  triplet loss / ArcFace loss for metric learning.
- Dataset is small (30 identities) and does not represent real-world in-the-wild
  illumination variation.
- Future: Qwen-VL end-to-end prompting, real (non-synthetic) yin-yang data,
  larger-scale evaluation.

## Citation

```
Zheng et al., 2026 — YinYangFace: Qwen-Perceptor Guided Relighting
for Half-Face Illumination. Tech report (V2).
```
