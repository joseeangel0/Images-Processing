# 📊 Final Project Report: Image Processing Filtering Benchmark

**Execution Timestamp:** 2026-03-10 06:29:38
**Image Resolution:** 1024x683 pixels
**Total Pixels Processed:** 699,392 pixels

## 1. Architectural Implementation Overview

This project evaluates the performance disparities across three distinct computational architectures, tasked with generating mathematically identical Convolutional Image Processing Filters.

### 🔬 Methodology Breakdown

- **Pure Python (`src/filters_python.py`)**: 
  - **Architecture:** Standard CPython interpretation using high-level list objects.
  - **Constraints:** Heavy overhead from the Global Interpreter Lock (GIL), dynamic type-checking, and sequential execution.
  - **Insight:** Useful as a baseline to demonstrate the performance floor of non-optimized code.

- **NumPy (`src/filters_numpy.py`)**: 
  - **Architecture:** Vectorized matrix operations using pre-compiled C/Fortran backends.
  - **Optimization:** Leverages SIMD (Single Instruction, Multiple Data) instructions and contiguous memory access.
  - **Insight:** The "Industry Standard" for balancing developer productivity with execution speed.

- **NumPy + Cython (`src/filters_cython.pyx`)**: 
  - **Architecture:** Strictly-typed C-extensions compiled Ahead-of-Time (AOT).
  - **Optimization:** Eliminates Python interpreter overhead entirely within the loops. Uses direct memory-view pointer arithmetic.
  - **Insight:** Represents the performance ceiling for CPU-bound tasks in Python.

## 2. Mathematical Context of Filters

Each filter algorithm serves a unique spatial transformation purpose:
1. **Gaussian Blur:** Uses a 2D Gaussian function kernel to smooth noise by calculating a weighted average of surrounding pixels.
2. **Sobel Operator:** Calculates the image gradient intensity at each pixel, effectively highlighting edges by finding vertical and horizontal derivatives.
3. **Median Filter:** A non-linear filter that replaces each pixel with the median value of its neighborhood, exceptionally effective at removing 'salt-and-pepper' noise while preserving edges.

## 3. Performance Analysis & Telemetry

### 3.1. Benchmark Execution Summary

| Filter Algorithm | Pure Python | NumPy Vectorized | Cython Optimized | Acceleration (Py → Cy) |
|------------------|-------------|------------------|------------------|-------------------|
| **Gaussian (Blur)** | 1.5665    s | 0.0338    s | 0.0041    s | **~386.2x** |
| **Sobel (Edges)**    | 2.3490    s | 0.0558    s | 0.0102    s | **~231.1x** |
| **Median (Noise)**   | 1.4968    s | 0.0656    s | 0.0448    s | **~33.4x** |

**Visual Performance Graph (Logarithmic Scale):**

<p align="center">
  <img src="../images/output/performance_chart.png" alt="Performance Chart" width="800">
</p>

### 3.2. Deep Dive Insights

1. **The Python Loop Problem:** The pure Python implementation is bottlenecked by object creation and dictionary lookups per pixel. For this image, it performs ~6,294,528 operations inside nested loops, leading to orders-of-magnitude slowdowns.
2. **Vectorization vs. Iteration:** NumPy is remarkably fast but requires temporary array allocations for sliding windows. Cython bypasses this by modifying memory in-place or via direct pointers, which is why it often outperforms NumPy in fixed-kernel convolutions.
3. **Hardware Considerations:** These results are bound by single-core CPU frequency. Modern CPUs with larger L2/L3 caches will show even better Cython performance due to improved memory locality during pointer arithmetic.

## 4. Visual Results & Fidelity Verification

Despite the computational differences, the output images are mathematically consistent across all implementations.

### 4.1. Output Samples
- **Gaussian Blur:** Effectively eliminated high-frequency noise.
- **Sobel Edges:** Captured structural contours with high precision.
- **Median Filter:** Best balance between noise reduction and edge preservation.

<p align="center">
  <img src="../images/output/Cython_comparison.png" alt="Cython Filter Results" width="900">
  <br>*(Visualized: Cython-accelerated output comparison)*
</p>

## 5. Future Scalability & Recommendations

To further reduce execution time, the following optimizations are recommended:
- **Parallelization (OpenMP):** Implementing `prange` in Cython to utilize all CPU cores.
- **GPU Acceleration (CUDA/CuPy):** Offloading pixel math to thousands of GPU cores for real-time video processing.
- **Cache Locality:** Optimizing the memory access pattern to be more "Cache-Friendly" (Row-major vs Column-major).
