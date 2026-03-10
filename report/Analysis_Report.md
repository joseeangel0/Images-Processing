# Final Project Report: Image Processing Filters

**Note**: The image processed in this execution had a resolution of 1024x683 pixels.

## 1. Code Implementation

The project is structured logically across specific Python modules, abstracting three different image processing filters (Gaussian Blur, Sobel Edge Detection, and Median Noise Reduction):

- **Pure Python (`src/filters_python.py`)**: Well-structured pure Python implementations relying strictly on nested `for` loops and array indexing. Contains `apply_gaussian_python()`, `apply_sobel_python()`, and `apply_median_python()`.
- **NumPy (`src/filters_numpy.py`)**: Vectorized matrix operations using NumPy core logic to calculate overlapping strides dynamically without using explicit loops. Contains `apply_gaussian_numpy()`, `apply_sobel_numpy()`, and `apply_median_numpy()`.
- **NumPy + Cython (`src/filters_cython.pyx`)**: Compiled strictly typed C-extensions disabling bounds checking and utilizing native C data-types for nested looping algorithms. Contains `apply_gaussian_cython()`, `apply_sobel_cython()`, and `apply_median_cython()`.

## 2. Performance Analysis

All execution models were subjected to identical image arrays dynamically loaded from the source folder.

### 2.1. Results Comparison Summary
Here is a comparison of the execution times for applying the filters on the sample image:

| Filter   | Pure Python | NumPy Vectorized | Cython Optimized |
|----------|-----------------|-----------|------------|
| **Gaussian** | 1.2798         s | 0.0168   s | 0.0043    s |
| **Sobel**    | 1.9968         s | 0.0406   s | 0.0071    s |
| **Median**   | 1.2047         s | 0.0554   s | 0.0396    s |

**Visual Performance Graph (Logarithmic Scale):**

<p align="center">
  <img src="../images/output/performance_chart.png" alt="Performance Chart" width="800">
</p>

### 2.2. Insights on Computational Optimization

1. **Pure Python vs NumPy:** The pure Python execution is by far the slowest approach. Python acts as an interpreter, dynamically checking variables, creating objects overhead on every iteration, and translating bytecode sequentially. NumPy optimizes this by translating our logic down to pre-compiled C routines that execute vectorized array math seamlessly without sequential looping bottlenecks. We see an extreme improvement in execution times using NumPy.
2. **NumPy vs Cython:** Although NumPy is heavily optimized, it carries memory allocations inside its generic backend due to standardizing array windows. Cython takes this a massive step further—we instruct Cython exactly how variables behave via strictly-typed syntax. The Gaussian filter convolution demonstrates the power of direct compile-time optimization in C as it runs dramatically faster than even our NumPy backend.

## 3. Visual Results

The output visual deliverables are mapped securely to the `/images/output` directory.
The pipeline ran effectively on the source image, capturing visual representations side-by-side using `matplotlib` showing the exact visual transformations derived by each filter computation technique.

- **Original Image:** Kept for visual reference across all benchmarks.
- **Gaussian Filter:** The overall spatial noise of the image and fine edge elements are appropriately smoothed.
- **Sobel Filter:** Horizontal and vertical variations detect structural borders translating the overall matrix to highlighting pure edge detections effectively.
- **Median Filter:** Distinct localized noise is dropped replacing localized structures with neighboring common denominators.

### 3.1 Python Output
<p align="center">
  <img src="../images/output/Python_comparison.png" alt="Python Visual Results" width="800">
</p>

### 3.2 NumPy Output
<p align="center">
  <img src="../images/output/NumPy_comparison.png" alt="NumPy Visual Results" width="800">
</p>

### 3.3 Cython Output
<p align="center">
  <img src="../images/output/Cython_comparison.png" alt="Cython Visual Results" width="800">
</p>
