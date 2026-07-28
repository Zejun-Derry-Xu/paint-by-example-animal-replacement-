# Precise Animal Replacement System
### Enhanced *Paint-by-Example* with LoRA Fine-tuning and CLIP Pose Retrieval

<p align="center">
  <img src="https://img.shields.io/badge/Base-Paint--by--Example-blue?style=for-the-badge" alt="Base Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Diffusers-red?style=for-the-badge&logo=pytorch" alt="Framework">
  <img src="https://img.shields.io/badge/Methods-LoRA%20%7C%20CLIP-orange?style=for-the-badge" alt="Methods">
</p>

> **🛑 Note on Source Code Access**
> To comply with academic integrity policies, the full source code is hosted in a **private repository**.
> This page is a **Portfolio Showcase** documenting the system design, method, and evaluation results. I'm happy to walk through the implementation and share specific code snippets during the interview process. **Feel free to reach out for access.**

---

## 🎯 The Mission: "Understand the animal, don't copy-paste it."

Generic image-editing models struggle with **species-specific detail** (fur texture, eye shape, breed characteristics), and exemplar-based methods tend to **literally copy-paste** the reference into the scene.

**Goal:** high-fidelity animal replacement — insert a *reference* animal into a *masked region* of a source image, preserving pose and background, **without copy-paste artifacts**.

We built on **Paint-by-Example** (exemplar-based latent diffusion), whose *information bottleneck* already discourages pixel-level copying, and added **two components** on top of it: domain-specific **LoRA fine-tuning** and **CLIP-based pose retrieval**.

---

## 🏗️ System Overview

### 1️⃣ LoRA Fine-tuning — Recovering Species Detail *(our addition)*
Parameter-efficient LoRA adaptation (`r=16`, `α=32`) of the UNet's attention projections (`to_q / to_k / to_v / to_out`) on a cat/dog replacement set, restoring fur texture, facial and eye features the base model tends to wash out. The VAE and reference encoder stay **frozen**, so training is lightweight and stable.

### 2️⃣ CLIP Pose Retrieval — Reducing Structural Distortion *(our addition)*
Instead of feeding an arbitrary reference, we crop the region to be replaced, embed it with **CLIP ViT-B/32**, and retrieve the **pose-closest** candidate from a purpose-built reference bank by cosine similarity — a **cross-species** query (dog region → cat bank). Generation therefore starts from a reference already aligned to the target pose, which visibly reduces warping.

### 3️⃣ Information Bottleneck — Why It Doesn't Copy-Paste *(inherited from Paint-by-Example)*
Before fusion, the reference image is compressed by the pretrained reference encoder into a **single semantic feature vector**. This bottleneck is what stops the model from learning an identity mapping (pixel copy-paste) and forces it to capture the reference's **semantics**. We did not modify this mechanism — understanding it is what motivated design choices 1️⃣ and 2️⃣.

---

## ⚙️ How It Works

Training follows the Stable-Diffusion / Paint-by-Example noise-prediction objective:

- The ground-truth image is encoded to a **VAE latent** and noised.
- The UNet receives **noisy latent + masked source latent + mask + reference embedding**.
- The reference image passes through the **pretrained encoder → information bottleneck → compressed feature vector**.
- **LoRA** adapts the attention layers; the pretrained backbone stays frozen.
- Optimized with an **MSE noise-prediction loss**, fp16 + gradient scaling.

<img width="517" height="288" alt="截屏2026-07-24 下午12 32 54" src="https://github.com/user-attachments/assets/4dfd718f-11f1-4d98-9da1-77a6ddfe010e" />

### Pipeline at a glance

| Phase | What happens |
|---|---|
| **1 · Data** | Build `(ground truth, masked source, reference, mask)` samples from **MS-COCO 2017** cat/dog images; validate every file path before training |
| **2 · LoRA training** | 8 epochs, `lr=5e-5`, 512×512, fp16 + `GradScaler`, per-epoch checkpointing (A100) |
| **3 · Reference bank** | Extract COCO cat instances → **412-image candidate bank** → precompute CLIP visual features into a retrieval pool |
| **4 · Inference & ablation** | Cross-species CLIP pose retrieval → 50-step sampling → 2×2 ablation grid |

---

## 🏆 Results — 2×2 Ablation

We compared four configurations: **`{Base, LoRA} × {Random-Ref, CLIP-Ref}`**.

- **Ours (LoRA + CLIP)** produced the most natural results — correct species detail **and** pose alignment.
- **No LoRA** → weaker species-specific detail; **random reference** → structural distortion or copy-paste artifacts.

We also placed our output **side by side with SOTA closed models (Gemini, GPT-Image)**: a gap remains, but this lightweight, fully-controllable pipeline stays visually competitive on the task.

<img width="988" height="415" alt="截屏2026-07-24 下午12 11 34_副本" src="https://github.com/user-attachments/assets/d3d94acd-9e8f-4018-95f6-711466faf6fb" />

> **Scope of evaluation:** results reported here are **qualitative** (side-by-side comparison on held-out samples). A batch FID / LPIPS harness on COCOEE is scaffolded in the codebase but was not run within the course timeline — that is the natural next step.

---

## 🚧 Limitations & Next Steps
- Quantitative evaluation (FID / LPIPS on COCOEE) is scaffolded but not yet executed.
- The reference bank is restricted to COCO cats; broader species coverage would test generality.
- CLIP image embeddings capture pose only implicitly — an explicit pose/keypoint signal would likely retrieve better references.

---

## 🧰 Tech Stack
`PyTorch` · `diffusers` · `Paint-by-Example` · `PEFT / LoRA` · `CLIP ViT-B/32` · `MS-COCO 2017`

---

## 👥 Team & Contributions
A 3-person graduate course project (Georgetown University, Computer Vision, Spring 2026): **Zejun Xu, Yiding Zhu, Yifan Hu** — jointly responsible for paper research, method selection, implementation, model training, ablation, failure analysis, and evaluation.

📩 *For source-code access or a technical walk-through, feel free to contact me.*
