# 🖼️ Image Processing Benchmarking Project

A high-performance benchmark suite designed to evaluate the execution disparity between different computational architectures when applying fundamental image processing filters.

## 🚀 Key Highlights

- **Performance Benchmarking:** Comparative analysis of **Pure Python**, **NumPy**, and **Cython**.
- **Hybrid Implementation:** Utilizing C-extensions for maximum pixel-processing efficiency.
- **Visual Analytics:** Automatic generation of performance charts and side-by-side filter comparisons.
- **Dynamic Reporting:** Real-time generation of advanced technical reports based on benchmark telemetry.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Numerical Computing:** [NumPy](https://numpy.org/) for vectorized operations.
- **Acceleration:** [Cython](https://cython.org/) for C-compiled performance.
- **Visualization:** [Matplotlib](https://matplotlib.org/) for analytics and [OpenCV](https://opencv.org/) for image I/O.

---

## 📂 Project Structure

- `src/`: Core source logic.
  - `main.py`: Orchestrator for benchmarking and report generation.
  - `filters_python.py`: Naive implementations (Baseline).
  - `filters_numpy.py`: Vectorized optimizations.
  - `filters_cython.pyx`: Heavily optimized C-compiled endpoints.
  - `setup.py`: Compilation configuration for Cython modules.
- `report/`: Dynamically generated technical analysis.
- `images/`: Input source data and generated visual results.

---

## ⚙️ Setup & Execution

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Compile Optimization Layer (Required)

The Cython extensions must be compiled locally to match your architecture:

```bash
cd src
python setup.py build_ext --inplace
```

### 3. Run the Suite

Execute the pipeline to process images, run benchmarks, and generate the final report:

```bash
python main.py
```

Results and visual comparisons will be saved under `images/output/`, and the deep-dive analysis will be updated in `report/Analysis_Report.md`.
