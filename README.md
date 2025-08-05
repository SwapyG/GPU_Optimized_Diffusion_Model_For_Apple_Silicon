# GPU-Optimized Diffusion Model for Apple Silicon

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![CoreML](https://img.shields.io/badge/CoreML-black?logo=apple&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete implementation of a Diffusion Model for image generation, ported from a CUDA-based architecture to run efficiently on Apple Silicon (M1/M2/M3). This project showcases a multi-tiered optimization strategy, including a custom C++ Metal kernel and a Core ML-accelerated pipeline.

The core of this project was not just the implementation, but the rigorous debugging of low-level memory corruption bugs (`segmentation fault`, `malloc`) that arose from interfacing PyTorch with custom Metal code, culminating in a stable, robust, and high-performance application.

## Key Features

-   **Multi-Tiered Optimization:** Compares the performance of three different backends:
    1.  **PyTorch MPS:** The standard, baseline performance on Apple Silicon GPUs.
    2.  **Custom Metal Kernel:** A hand-written Metal kernel in Objective-C++ to fuse operations, demonstrating low-level GPU programming.
    3.  **Core ML:** A fully accelerated pipeline using Apple's Core ML framework to leverage the Apple Neural Engine (ANE) for maximum inference speed.
-   **Full Training & Sampling Pipeline:** Includes scripts to train the model from scratch and generate images.
-   **Advanced Debugging:** A practical demonstration of solving complex, real-world memory management issues between Python, PyTorch C++, and Metal.

## Performance Benchmarks

The following benchmarks were recorded on an [Your Mac Model, e.g., M1 Pro] generating 16 MNIST samples after 20 epochs of training.

| Implementation | Time to Generate 16 Images | Speed vs. Baseline | Key Technology |
| :--- | :---: | :---: | :--- |
| `original` (PyTorch MPS) | ~11-12s | 1x | Standard GPU |
| `metal` (Custom Kernel) | ~17s | ~0.7x | Custom C++ & Metal |
| `coreml` (Neural Engine) | **~5-6s** | **~2x** | Apple Neural Engine |

*Note: The custom `metal` kernel is slightly slower due to the necessary "CPU Bridge" to ensure stability. This highlights a real-world trade-off between pure GPU optimization and robust API integration.*

## The Engineering Journey: Debugging Low-Level Bugs

A significant part of this project was overcoming critical, low-level bugs. This demonstrates a deep understanding of the full stack, from Python down to the metal.

-   **Solved `malloc: Incorrect checksum` & `segmentation fault`:** Traced a memory corruption bug caused by a compiler optimization race condition. Fixed by recompiling the C++ extension with a safer optimization flag (`-O1`) and implementing a robust `@autoreleasepool`.
-   **Solved `RuntimeError: Backend doesn't support getDeviceFromPtr()`:** Identified and fixed a limitation in PyTorch's C++ API for the MPS backend by creating a "CPU Bridge" to safely transfer tensor data to and from the custom Metal function.
-   **Solved `RuntimeError: value type not convertible`:** Debugged a data type mismatch in the Core ML pipeline, fixed by explicitly casting the timestep tensor to `float32` before prediction.

## Installation

### Prerequisites

-   An Apple Silicon Mac (M1, M2, M3, etc.).
-   macOS 12.0 or later.
-   Xcode Command Line Tools: `xcode-select --install`

### Setup

```bash
# 1. Create and activate a fresh conda environment
conda create -n diffusion-metal python=3.10
conda activate diffusion-metal

# 2. Install Python dependencies
pip install torch torchvision
pip install matplotlib numpy coremltools

# 3. Build the custom Metal C++ extension
python setup.py install```

## Usage

### 1. Train the Model

You must train a model first. The trained weights will be saved to `diffusion_model.pt`. Training for 20+ epochs is recommended for good image quality.

```bash
python main.py --mode train --n_epochs 20
```

### 2. Generate Images

Once the model is trained, you can generate samples using any of the three implementations.

```bash
# Run the baseline PyTorch MPS implementation
python main.py --mode sample --implementation original

# Run the stable custom Metal kernel implementation
python main.py --mode sample --implementation metal

# Run the high-performance Core ML implementation
python main.py --mode sample --implementation coreml
```
Generated images will be saved in the `samples/` directory.````
