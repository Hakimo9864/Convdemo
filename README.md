# Convolution Demo

## Overview
This repository explores **convolutions** and **kernels** in image processing. Convolutions are fundamental operations used in computer vision and deep learning to extract features from images through sliding window operations.

## What Are Convolutions?
A convolution applies a small matrix (kernel) to an image by sliding it across each pixel and computing weighted sums. This process helps detect patterns such as edges, textures, and shapes.

## What Are Kernels?
Kernels are small matrices used in convolution operations. Each kernel is designed to detect specific features:
- **Blur Kernel**: Smooths an image by averaging neighboring pixels
- **Horizontal Edge Kernel**: Detects horizontal edges and transitions
- **Vertical Edge Kernel**: Detects vertical edges and transitions

## How to Use
1. Ensure you have the required dependencies installed (see `requirements.txt`, this is easily done using `python -m pip install -r requirements.txt`
2. Place an image file at `data/kitty.jpg` or modify the file path in `main.py`
3. Run the script:
   ```bash
   python main.py
   ```
4. The script will display the original grayscale image and the results of applying various kernels

## Learning Objectives
- Understand how 2D convolutions work on images
- Explore different kernel designs and their effects
- Visualize feature detection in image processing
