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

**Goal:** high-fidelity animal replacement — insert a *reference* animal into a *masked region* of a source image, preserving pose and background, **without copy-paste artifacts**. We built on **Paint-by-Example** (exemplar-based latent diffusion) and improved it along three axes.

---

## 🏗️ System Overview

### 1️⃣ LoRA Fine-tuning — Recovering Species Detail
Parameter-efficient LoRA adaptation of the diffusion model's attention layers on breed-specific cat/dog data, restoring fur texture, facial and eye features the base model tends to wash out — while keeping the pretrained backbone frozen and training lightweight.

### 2️⃣ CLIP Pose Retrieval — Reducing Structural Distortion
For the region to be replaced, we compute **CLIP embeddings** over candidate references and pick the **pose-closest** one by cosine similarity, so generation starts from a reference already aligned to the target pose — minimizing warping and structural distortion.

### 3️⃣ Information Bottleneck — Forcing Semantics over Copying
The reference is passed through a **compressed feature injection (information bottleneck)** before fusion, which prevents the model from learning an identity mapping (pixel copy-paste) and forces it to capture the reference's **semantics** instead.

---

## ⚙️ How It Works

Training follows the Stable-Diffusion / Paint-by-Example noise-prediction objective:

- The ground-truth image is encoded to a **VAE latent** and noised.
- The UNet receives **noisy latent + source latent + mask + reference embedding**.
- The reference image passes through a **pretrained encoder → information bottleneck → compressed feature vector**.
- **LoRA** adapts the attention layers; the pretrained model stays frozen for stable, lightweight training.
- Optimized with an **MSE noise-prediction loss**.


<img width="517" height="288" alt="截屏2026-07-24 下午12 32 54" src="https://github.com/user-attachments/assets/4dfd718f-11f1-4d98-9da1-77a6ddfe010e" />

---

## 🏆 Results — 2×2 Ablation

We evaluated across a **`{Base, LoRA} × {Random-Ref, CLIP-Ref}`** ablation:

- **Ours (LoRA + CLIP)** produced the most natural results — correct species detail **and** pose alignment.
- **No LoRA** → weaker species-specific detail; **Random reference** → structural distortion or copy-paste artifacts.

We also benchmarked against **SOTA closed models (Gemini, GPT-Image)**: a gap remains, but our lightweight, fully-controllable pipeline stays competitive on this task.

<img width="988" height="415" alt="截屏2026-07-24 下午12 11 34_副本" src="https://github.com/user-attachments/assets/d3d94acd-9e8f-4018-95f6-711466faf6fb" />


---

## 🧰 Tech Stack
`PyTorch` · `diffusers` · `Paint-by-Example` · `LoRA` · `CLIP` · `MS-COCO`

---

## 👥 Team & Contributions
A 3-person graduate course project (Georgetown University, CV, Spring 2026): **Zejun Xu, Yiding Zhu, Yifan Hu** — jointly responsible for paper research, method selection, implementation, model training, ablation, failure analysis, and evaluation.

📩 *For source-code access or a technical walk-through, feel free to contact me.*
