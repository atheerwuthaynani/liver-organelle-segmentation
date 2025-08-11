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
- `liver_segmentation.py` - Main segmentation pipeline script
- `results/` - Performance metrics and visualizations
- `requirements.txt` - Required Python packages

## How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Update data paths in `configure_data_paths()` function
3. Run the pipeline: `python liver_segmentation.py`
4. Results will be generated automatically in `/content/predictions/`
