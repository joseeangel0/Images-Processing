# Image Processing Benchmarking Project

This project implements and compares the runtime execution of common image filters (Gaussian, Sobel, Median) using Pure Python, NumPy, and Cython.

## Project Structure

- `src/`: Source code.
  - `main.py`: Main executable file to run and benchmark the algorithms.
  - `filters_python.py`: Implementations relying purely on standard Python semantics.
  - `filters_numpy.py`: Optimizations using purely NumPy vectorization.
  - `filters_cython.pyx`: Heavily optimized compiled endpoints in C.
  - `setup.py`: Cython build configuration.
- `report/`: Analysis on runtime execution results.
- `images/`: The source and results data.

## Setup Instructions

**1. Install Python Dependencies**
Ensure you have a Python `3.x` environment setup.

```bash
pip install -r requirements.txt
```

**2. Compile Cython Extentisions**
You must compile the Cython file prior to execution:

```bash
cd src
python setup.py build_ext --inplace
```

**3. Run the Benchmark**
With a sample image located inside `images/input/sample.jpg`, execute the pipeline:

```bash
python main.py
```

Results are saved under `images/output/`.
