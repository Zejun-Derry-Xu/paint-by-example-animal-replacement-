# Precise Animal Replacement System
### Exemplar-based Image Editing with Diffusion Models — Enhancing *Paint-by-Example* with LoRA, CLIP Pose Retrieval & a Tunable Information Bottleneck

<p align="center">
  <img src="https://img.shields.io/badge/Base-Paint--by--Example-blue?style=for-the-badge" alt="Base Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Diffusers-red?style=for-the-badge&logo=pytorch" alt="Framework">
  <img src="https://img.shields.io/badge/Conditioning-CLIP%20%7C%20Info--Bottleneck-orange?style=for-the-badge" alt="Conditioning">
</p>

> **🛑 Note on Source Code Access**
> To comply with academic integrity policies, the full source code for this project is hosted in a **private repository**.
> This page is a **Portfolio Showcase** — it documents the system design, the key technical contribution, and the evaluation results. I'm happy to walk through the implementation and share specific code snippets during the interview process. **Feel free to reach out for access.**

---

## 🎯 The Mission: "Understand the animal, don't copy-paste it."

Generic image-editing models struggle with **species-specific detail** (fur texture, eye shape, breed characteristics), and exemplar-based methods tend to **literally copy-paste** the reference into the scene.

**Our goal:** high-fidelity animal replacement — insert a *reference* animal into a *masked region* of a source image, preserving pose and background, **without copy-paste artifacts**.

We built our system on top of **Paint-by-Example** (exemplar-based latent diffusion) and improved it along three axes.

---

## 🏗️ System Overview

### 1️⃣ LoRA Fine-tuning — Recovering Species Detail
Parameter-efficient fine-tuning on breed-specific cat/dog imagery to restore fur texture, facial and eye features that the base model tends to wash out.

### 2️⃣ CLIP Pose Retrieval — Reducing Structural Distortion
Given the region to be replaced, we compute **CLIP embeddings** over candidate references and select the **pose-closest** one by cosine similarity — so the generator starts from a reference that already matches the target pose, minimizing warping.

### 3️⃣ Tunable Information Bottleneck — Forcing Semantics over Copying *(core contribution)*
The heart of the project. See below.

---

## 🔬 Technical Deep-Dive: The Adjustable Information Bottleneck

The original Paint-by-Example compresses the reference image into a **single 1×1024 token** before it enters the UNet cross-attention. That aggressive bottleneck is *why* the model learns semantics instead of pixel copying.

**We turned that fixed bottleneck into a controlled experimental variable.** We implemented a custom conditioning encoder that:

- keeps the pretrained **CLIP vision backbone** and the author's original mapper intact;
- inserts a **reconfigurable `tokens × dim` bottleneck** (a projection + learnable positional tokens + an optional Transformer block + LayerNorm + output projection);
- is **initialized from the pretrained weights** (identity-style init) so it starts as close to the original model as possible;
- is **parameter-efficient**: VAE, UNet and CLIP backbone are frozen — only the new bottleneck is trained.

We then ran a systematic study across bottleneck shapes:

| Config (`tokens × dim`) | Intent |
|---|---|
| `1 × 1024` | Paper baseline |
| `1 × 512`  | Tighter bottleneck |
| `2 × 512`  | Same capacity, more tokens |
| `4 × 256`  | Further token split |
| `2 × 1024` | Loosened bottleneck |

**Training setup:** epsilon (noise-prediction) MSE objective, DDPM scheduler, classifier-free guidance via conditioning dropout, mixed-precision (fp16), gradient accumulation.

> **Key insight:** the bottleneck's *shape*, not just its size, controls the trade-off between **semantic understanding** and **identity copying** — too loose and the model copy-pastes; too tight and it loses the reference identity.




---

## 🏆 Results — 2×2 Ablation

We evaluated the full pipeline across a **`{Base, LoRA} × {Random-Ref, CLIP-Ref}`** ablation:

- **Ours (LoRA + CLIP)** produced the most natural results — correct species detail *and* pose alignment.
- **No LoRA** → weaker species-specific detail; **Random reference** → structural distortion or copy-paste artifacts.

We also benchmarked against **SOTA closed models (Gemini, GPT-Image)**: a gap remains, but our lightweight, fully-controllable pipeline stays competitive on this task.

<img width="988" height="415" alt="截屏2026-07-24 下午12 11 34_副本" src="https://github.com/user-attachments/assets/15aead8b-efa2-4222-9d31-8b30af51a547" />

---

## 🧰 Tech Stack
`PyTorch` · `diffusers` · `transformers` · `Paint-by-Example (Fantasy-Studio)` · `CLIP` · `OpenImages / MS-COCO`

---

## 👥 Team & Contributions
A 3-person graduate course project (Georgetown University, CV, Spring 2026): **Zejun Xu, Yiding Zhu, Yifan Hu** — jointly responsible for paper research, method selection, implementation, model training, the bottleneck ablation, failure analysis, and evaluation.

📩 *For source-code access or a technical walk-through, feel free to contact me.*
