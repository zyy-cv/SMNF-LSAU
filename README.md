# Arbitrary-Scale Image Super-Resolution via Scale-aware Multiscale Nonlocal Feature Extraction


This repository contains the core architectural implementation of our proposed arbitrary-scale super-resolution network. Our method integrates a **Scale-aware Multiscale Nonlocal Feature (SMNF)** extraction module and a **Local Structure-Adaptive Upsampling (LSAU)** module into standard backbones (EDSR / RDN) to achieve continuous magnification with high fidelity.

---

## 🚀 Core Architecture & Innovations

Our code is currently streamlined to highlight the core contributions requested during the review process. The full training and evaluation scripts will be released upon paper acceptance.

  
## 📦 Pre-trained Models

You can download our pre-trained weights to reproduce the results reported in the paper.

* **Baidu Netdisk (百度网盘):** `[Link to be added here]` (Extraction code: `xxxx`)
* **Google Drive:** `[Link to be added here]`

*Note: Please place the downloaded `.pth` files into the `./weights/` directory.*

---

## 💻 Environment Setup

The code was developed and tested with the following specifications:
* Python 3.7
* PyTorch 1.10
* CUDA 11.3

---

## 📝 Acknowledgements
This project is built upon the foundation of several excellent open-source works, including [LIIF](https://github.com/yinboc/liif), [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch), and [CrossFormer](https://github.com/cheerss/CrossFormer). We thank the original authors for their contributions to the community.
