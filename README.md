---
# Brain MRI tumour classifier

![CI](https://github.com/keerthisuresh06/mri-classifiers/actions/workflows/ci.yml/badge.svg)

> **Research and educational project. Not a medical device. Do not use for diagnosis.**

Classifies a 2D brain MRI slice as glioma, meningioma, pituitary tumour or no tumour,
with a fine-tuned ResNet18, served through FastAPI, packaged with Docker and tested in CI.

- **Live demo:** https://huggingface.co/spaces/Keerthisuresh12/mri-classifier-demo
- **Model card:** https://huggingface.co/Keerthisuresh12/mri-resnet18

## Results (held-out test set, used once)

| Model | Accuracy |
|---|---|
| From-scratch CNN baseline | 85.0% |
| ResNet18, fine-tuned | **95.5%** |

Calibrated with temperature scaling (ECE 0.026 → 0.010). Full per-class metrics
and limitations are in the [model card](https://huggingface.co/Keerthisuresh12/mri-resnet18).

## What I found along the way

- **The official Kaggle split leaks.** 294 groups of near-duplicate images crossed
  train and test. I pooled the data and re-split by perceptual hash.
- **BatchNorm made validation unstable.** Controlled experiments traced erratic
  validation loss to BatchNorm's evaluation statistics; GroupNorm fixed it.
- **Possible shortcut.** The "no tumour" class comes from a different source with
  smaller RGB images. 16 of 17 missed tumours were RGB images. A resolution-only
  test did not change predictions; other source traits remain untested.
- **Confidently wrong out of distribution.** A CT-looking scan was called
  "no tumour" with 98.9% confidence.
- **My first Grad-CAM maps were a bug.** A duplicated hook produced identical
  heat maps for every image. A sanity check (class-specific maps, random-weight
  baseline) exposed it.

## Architecture

    Kaggle training ──> HF model repo (weights, pinned revision)
                              │
          ┌───────────────────┴───────────────────┐
     FastAPI service + Docker              Gradio demo (HF Space)
     (/predict, /health, /demo)            same predict() code
     tested in GitHub Actions CI

## Run locally

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    python -m pytest -v
    uvicorn app.main:app --reload        # http://127.0.0.1:8000

With Docker:

    docker build -t mri-classifier .
    docker run --rm -p 8000:8000 mri-classifier

Weights download automatically from the Hugging Face model repo on first use.

## Future work

- Occlusion tests on missed tumours
- External validation on MRI from a different source
---