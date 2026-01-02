# Environment Setup

This document describes how to set up a development and training environment for **TableSense2**. The project supports local GPU workstations, cloud notebooks, and containerized execution.

---

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 or higher
- **Core libraries**:
  - `openpyxl` (≥3.1.0)
  - `pandas` (≥2.0.0)
  - `numpy` (≥1.24.0, <2.0)
  - `scikit-learn` (≥1.3.0)
  - `scipy` (≥1.10.0)
  - `pyyaml` (≥6.0)
  - `tqdm` (≥4.65.0)
- **Experiment tracking (optional)**: `wandb` (≥0.15.0)

---

## Option A — Conda (recommended)

This option most closely matches the reference development environment and is recommended for GPU users.

```bash
conda env create -f environment.yml
conda activate tablesense2
pip install -e .
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate tablesense2
pip install -e .
```

---

## Option B — Python venv + pip

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install -e .
```

---

## GPU Notes

- CUDA-enabled GPUs are optional but strongly recommended for training
- PyTorch will automatically fall back to CPU if CUDA is unavailable
- Verify GPU visibility:

```bash
python - <<EOF
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

---

## Preprocessing

Please make sure that you have run the preprocessing task defined in `docs/preprocess_tablesense_files.md`

---

## Verification

Run the test suite to verify installation:

```bash
pytest tests/ -v
```

Some tests require training data to be present in `training_data/`.

## Train the TableDetector

```bash
python scripts/train_detector.py --config configs/detector_config.yaml --project-root .
```

Or use the minimal config for faster development/testing:

```bash
python scripts/train_detector.py --config configs/detector_config_minimal.yaml --project-root .
```

## Apple’s MPS backend (Metal Performance Shaders) notes

On Apple silicon, PyTorch may still execute some ops on the CPU because they’re not implemented in the mps backend yet. If you set device="mps" in your config, set PYTORCH_ENABLE_MPS_FALLBACK=1 before launching training so that unsupported ops (e.g., grid_sampler_2d_backward) automatically run on the CPU instead of aborting. Expect this path to be slower than native MPS execution.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_detector.py --config configs/detector_config_minimal.yaml --project-root .
```


## Warnings during training

There are warnings that show up when loading files 