# Multi-Organelle Liver Segmentation using nnU-Net

## Project Overview
This project implements automated segmentation of liver organelles in electron microscopy images using nnU-Net framework.

**Institution:** King Abdullah University of Science and Technology (KAUST)  
**Event:** SUDS Showcase 2025  
**Author:** Atheer Alwuthaynani

## Results Summary
- **Best Performance:** Lipid Droplets (Dice: 0.806)
- **Most Challenging:** ER segmentation (Dice: 0.182)
- **Training Approach:** Single image training (proof of concept)
- **Prediction Success:** 126 images automatically segmented
- **3D Output:** Volume generated for ITK-SNAP
- **Technical Achievement:** Automated multi-organelle pipeline established

## Methodology
- **Framework:** nnU-Net v2 (self-configuring)
- **Training Data:** Single manually annotated image
- **Test Data:** 3 validated images with ground truth
- **Organelles:** Mitochondria, Nucleus, endoplasmic reticulum (ER), Lipid Droplets, Cell Boundaries

## Repository Contents
- `notebooks/` - Jupyter notebooks with complete pipeline
- `results/` - Performance metrics and visualizations
- `docs/` - Additional documentation

## How to Use
1. Open the Colab notebook
2. Follow the step-by-step pipeline
3. Results will be generated automatically
