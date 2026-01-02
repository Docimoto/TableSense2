# TableSense2

**Modern, Excel‑native table detection and understanding (Apache‑2.0)**

---

## Overview

**TableSense2** is an open‑source research and engineering project for **table detection in Excel spreadsheets**, designed to help **researchers, applied ML engineers, and enterprise IT teams** train and evaluate their own models on spreadsheet‑native data. 

While the primary focus of this project is table detection, the underlying **transformer-based detection architecture is intentionally general** and can be extended to identify **other structural and semantic patterns in Excel files**.

This project is inspired by the [**TableSense**](https://arxiv.org/pdf/2106.13500) research paper published by **Microsoft Research (2020)**. The original [**TableSense**](https://github.com/microsoft/TableSense) Github repository does not contain any code. **TableSense2 is a fully independent implementation** that modernizes the original ideas using contemporary vision architectures and provides the full code.

> **Disclaimer**  
> TableSense2 is **not affiliated with, endorsed by, or maintained by Microsoft**. All code in this repository is an original implementation released as open source.

---

## Why TableSense2

### Excel is still a dominant business document

In many B2B domains—logistics, finance, pricing, contracts, operations—**Excel files are an important data exchange format** between organizations. In my own industry experience in the logistics industry, in several projects **over 95% of inbound documents were Excel spreadsheets**.

Despite this reality, most document‑understanding research and tooling focuses on **PDFs and rasterized images**, leaving Excel‑centric workflows underserved. TableSense2 exists to address this gap.

---

## Design Principles

### 1. Excel‑native, not image‑first

Many vision models require Excel files to be rasterized before processing. While convenient, this approach **discards spreadsheet‑specific signals**, including:

- Cell‑level formatting (date, currency, font, etc.)
- Merged cells and layout structure
- Formulas vs literal values
- Explicit Excel table objects
- Hidden, locked, or computed cells

TableSense2 operates **directly on Excel cell grids**, preserving these features so models can learn from them.

---

### 2. Cell‑level featurization

Instead of treating a spreadsheet as a flat image, TableSense2 constructs a **structured feature tensor per cell**, combining:

- Layout and merge indicators
- Formatting and style flags
- Text and numeric statistics
- Excel object metadata

These features are especially valuable for information extraction from:
- Dense enterprise spreadsheets
- Large multi‑table worksheets
- Financial and operational documents

---

### 3. Reproducible, usable research

The original TableSense project did not release:
- training code
- a reference model
- or the full dataset described in the paper

While a small dataset was included, its annotations require additional work before being suitable for training. 

TableSense2 provides:
- a complete end‑to‑end pipeline
- The original TableSense data has been cleaned up as far as possible but requires further clean up of the annotations
- a framework designed for your own **private and domain‑specific datasets**

The goal is not just to reproduce results, but to **enable further research and practical adoption** by both researchers and enterprise IT teams.

---

### 4. Modernized architecture (2026‑ready)

Since 2020, when the TableSense paper was written, vision research has advanced significantly. TableSense2 reflects this by adopting a modern detection stack:

- **ConvNeXt V2 backbone**  
  A high‑performance, fully convolutional encoder with improved inductive biases and training stability.

- **DETR‑style transformer detection head (Deformable DETR)**  
  End‑to‑end set‑based table detection using bipartite matching, avoiding heuristic post‑processing pipelines.

This architecture supports:
- multiple tables per sheet
- variable sheet sizes
- principled training and evaluation

---

## What TableSense2 Provides

- End‑to‑end **Excel table detection pipeline**
- **Cell‑level featurization** extracted directly from `.xlsx` files
- ConvNeXt V2 + Deformable DETR reference implementation
- Training, evaluation, and inference scripts
- [SimpleTableSense2AnnotationUI](https://github.com/Docimoto/SimpleTableSense2AnnotationUI) — a lightweight annotation UI for generating and validating Excel table detection ground truth, designed to pair with TableSense2 training workflows
- YAML‑based configuration system
- Docker-based environment for reproducible training, enabling cloud execution in Google Colab, managed GPU instances. You can easily extend to other public clouds
- Support for local training on GPU-equipped workstations and laptops
- Built‑in experiment tracking using Weights & Biases
- A foundation you can extend with your own datasets and research ideas

---

## Intended Audience

TableSense2 is designed for:

- **Academic researchers** exploring spreadsheet understanding, document AI, or table detection
- **Applied ML engineers** working with structured business documents
- **Enterprise IT and data teams** building Excel‑centric AI pipelines

The project aims to bridge **research clarity** and **practical usability**.

---

## References

- Microsoft Research (2020): [*TableSense: Spreadsheet Table Detection with Convolutional Neural Networks*](https://arxiv.org/pdf/2106.13500)

This project is inspired by the paper but is an independent implementation.

---

## License

TableSense2 is released under the **Apache License, Version 2.0**.

This license allows commercial use, modification, and distribution, while providing an explicit patent grant and clear attribution requirements.

See the [LICENSE](license.md) and [NOTICE](notice.md) files for details.

---

## Getting Started

The sections below explain how to:

1. Install dependencies
2. Prepare datasets. Please see [this document](utils/preprocess_tablesense_files.md).  
3. Train baseline and DETR‑based models
4. Evaluate and visualize results

Please continue reading for setup instructions and usage examples.

---

## Project Structure (High‑Level)

```
TableSense2/
├── configs/        # YAML configuration files
├── data_io/        # Dataset loading and annotations
├── docs/           # Additional documentation
├── evaluation/     # Metrics and evaluators
├── features/       # Cell‑level featurization
├── inference/      # Inference and hybrid aggregation
├── models/         # ConvNeXt V2 + DETR architectures
├── scripts/        # CLI entry points
├── tests/          # Unit and integration tests
├── training/       # Trainers, configs, experiment logging
├── training_data/  # Training dataset(s)
├── utils/          # Utilities
└── vertex-ai       # Scripts to run training on Vertex AI
```
---

## Documentation

- Setup & training: [docs/setup.md](docs/setup.md)
- Experiments & tracking: [docs/experiment_tracking.md](docs/experiment_tracking.md)
- Cloud & Docker training: [vertex_ai/README.md](vertex_ai/README.md)
- Model architecture: [docs/architecture.md](docs/architecture.md)

---

## Contributing

Contributions are welcome.

If you are interested in improving the codebase, adding datasets, or experimenting with new architectures, please see [contributing.md](contributing.md) for guidelines.

---

## Project Vision

TableSense2 is released in the **spirit of open research and open collaboration**.

The hope is that:
- researchers can build reproducible spreadsheet models
- practitioners can adapt the system to real business data
- the community can collectively advance Excel‑centric document AI

