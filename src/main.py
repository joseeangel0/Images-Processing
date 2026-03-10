import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from filters_python import apply_gaussian_python, apply_sobel_python, apply_median_python
from filters_numpy import apply_gaussian_numpy, apply_sobel_numpy, apply_median_numpy
from filters_cython import apply_gaussian_cython, apply_sobel_cython, apply_median_cython

def get_image(input_dir):
    """Loads the first image found in the input directory in grayscale."""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            filepath = os.path.join(input_dir, filename)
            print(f"Found image: {filename}")
            return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
    # Create a dummy image for testing if no image is found
    dummy_path = os.path.join(input_dir, 'sample.jpg')
    print("No image found. Generating a dummy sample.jpg...")
    img = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    cv2.imwrite(dummy_path, img)
    return img

def display_and_save(original, filters_results, title, output_dir):
    """Saves the visual comparison of filters."""
    plt.figure(figsize=(15, 5))
    
    titles = ['Original', 'Gaussian', 'Sobel', 'Median']
    images = [original] + filters_results
    
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}_comparison.png"))
    plt.close()

def generate_performance_chart(t_py, t_np, t_cy, output_dir):
    """Generates and saves a bar chart comparing the execution times."""
    
    # We use a logarithmic scale because pure Python is excessively slower
    # and would dwarf the Cython/NumPy bars otherwise.
    
    labels = ['Gaussian', 'Sobel', 'Median']
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, t_py, width, label='Pure Python', color='salmon')
    rects2 = ax.bar(x, t_np, width, label='NumPy', color='skyblue')
    rects3 = ax.bar(x + width, t_cy, width, label='Cython', color='lightgreen')
    
    ax.set_ylabel('Execution Time (seconds) - Log Scale')
    ax.set_title('Performance Comparison of Image Processing Filters')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale('log') # Log scale to visualize the huge difference
    ax.legend(loc='upper right')
    
    # Add exact text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_chart.png')
    plt.savefig(chart_path)
    plt.close()

def generate_report(image_shape, t_py, t_np, t_cy, report_path):
    """Generates a dynamic markdown analysis report based on the execution times."""
    
    # Use relative paths from the report folder to the images folder
    # report is in report/, images are in images/output/
    # so relative path is ../images/output/
    chart_img = "../images/output/performance_chart.png"
    py_img = "../images/output/Python_comparison.png"
    np_img = "../images/output/NumPy_comparison.png"
    cy_img = "../images/output/Cython_comparison.png"
    
    report_content = f"""# Final Project Report: Image Processing Filters

**Note**: The image processed in this execution had a resolution of {image_shape[1]}x{image_shape[0]} pixels.

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
| **Gaussian** | {t_py[0]:<15.4f}s | {t_np[0]:<9.4f}s | {t_cy[0]:<10.4f}s |
| **Sobel**    | {t_py[1]:<15.4f}s | {t_np[1]:<9.4f}s | {t_cy[1]:<10.4f}s |
| **Median**   | {t_py[2]:<15.4f}s | {t_np[2]:<9.4f}s | {t_cy[2]:<10.4f}s |

**Visual Performance Graph (Logarithmic Scale):**

<p align="center">
  <img src="{chart_img}" alt="Performance Chart" width="800">
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
  <img src="{py_img}" alt="Python Visual Results" width="800">
</p>

### 3.2 NumPy Output
<p align="center">
  <img src="{np_img}" alt="NumPy Visual Results" width="800">
</p>

### 3.3 Cython Output
<p align="center">
  <img src="{cy_img}" alt="Cython Visual Results" width="800">
</p>
"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\nDynamic report generated successfully at: {report_path}")


def main():
    print("Executing visual filters pipeline...")
    input_dir = os.path.join('..', 'images', 'input')
    output_dir = os.path.join('..', 'images', 'output')
    report_path = os.path.join('..', 'report', 'Analysis_Report.md')
    os.makedirs(output_dir, exist_ok=True)
    
    image = get_image(input_dir)
    print(f"Loaded image of shape: {image.shape}")
    
    # Task 1: Display the original image before applying filters (save it individually)
    original_out_path = os.path.join(output_dir, 'original_image.png')
    cv2.imwrite(original_out_path, image)
    print(f"Original grayscale image saved to: {original_out_path}")
    
    image_list = image.tolist()
    
    # 1. Pure Python
    print("\n--- Running Pure Python Filters ---")
    start = time.time()
    g_py = apply_gaussian_python(image_list)
    t_g_py = time.time() - start
    
    start = time.time()
    s_py = apply_sobel_python(image_list)
    t_s_py = time.time() - start
    
    start = time.time()
    m_py = apply_median_python(image_list)
    t_m_py = time.time() - start
    print(f"Gaussian: {t_g_py:.4f}s | Sobel: {t_s_py:.4f}s | Median: {t_m_py:.4f}s")
    
    # 2. NumPy Vectorized
    print("\n--- Running NumPy Filters ---")
    start = time.time()
    g_np = apply_gaussian_numpy(image)
    t_g_np = time.time() - start
    
    start = time.time()
    s_np = apply_sobel_numpy(image)
    t_s_np = time.time() - start
    
    start = time.time()
    m_np = apply_median_numpy(image)
    t_m_np = time.time() - start
    print(f"Gaussian: {t_g_np:.4f}s | Sobel: {t_s_np:.4f}s | Median: {t_m_np:.4f}s")
    
    # 3. NumPy + Cython
    print("\n--- Running Cython Filters ---")
    start = time.time()
    g_cy = apply_gaussian_cython(image)
    t_g_cy = time.time() - start
    
    start = time.time()
    s_cy = apply_sobel_cython(image)
    t_s_cy = time.time() - start
    
    start = time.time()
    m_cy = apply_median_cython(image)
    t_m_cy = time.time() - start
    print(f"Gaussian: {t_g_cy:.4f}s | Sobel: {t_s_cy:.4f}s | Median: {t_m_cy:.4f}s")

    # Save output visualization
    print("\nSaving visualizations...")
    display_and_save(image, [np.array(g_py, dtype=np.uint8), np.array(s_py, dtype=np.uint8), np.array(m_py, dtype=np.uint8)], 'Python', output_dir)
    display_and_save(image, [g_np, s_np, m_np], 'NumPy', output_dir)
    display_and_save(image, [g_cy, s_cy, m_cy], 'Cython', output_dir)
    
    print("\n--- Execution Times Summary ---")
    print(f"{'Filter':<15} | {'Pure Python':<15} | {'NumPy':<15} | {'Cython':<15}")
    print("-" * 65)
    print(f"{'Gaussian':<15} | {t_g_py:<15.4f} | {t_g_np:<15.4f} | {t_g_cy:<15.4f}")
    print(f"{'Sobel':<15} | {t_s_py:<15.4f} | {t_s_np:<15.4f} | {t_s_cy:<15.4f}")
    print(f"{'Median':<15} | {t_m_py:<15.4f} | {t_m_np:<15.4f} | {t_m_cy:<15.4f}")

    # Generate the dynamic report
    t_py = (t_g_py, t_s_py, t_m_py)
    t_np = (t_g_np, t_s_np, t_m_np)
    t_cy = (t_g_cy, t_s_cy, t_m_cy)
    generate_performance_chart(t_py, t_np, t_cy, output_dir)
    generate_report(image.shape, t_py, t_np, t_cy, report_path)

if __name__ == '__main__':
    main()
