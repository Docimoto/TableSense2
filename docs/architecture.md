# Architecture Overview

This document provides a technical overview of the TableSense2 model architecture.

---

## High-Level Design

TableSense2 formulates spreadsheet table detection as a **set-based object detection problem** operating over Excel-native cell grids.

---

## Backbone: ConvNeXt V2

- Fully convolutional, multi-stage encoder
- Operates on cell-feature tensors (not pixels)
- Produces multi-scale feature maps (C2–C5)

ConvNeXt V2 provides strong inductive biases while remaining compatible with transformer heads.

---

## Detection Head: Deformable DETR

- Fixed set of object queries
- Multi-scale deformable attention
- Bipartite (Hungarian) matching during training
- Joint classification + box regression losses (L1 + IoU/GIoU)

This approach avoids heuristic post-processing and supports multiple tables per sheet.

---

## Excel-Specific Considerations

Excel Tables apply formatting via table styles rather than cell-level styles. Libraries such as `openpyxl` only expose cell-level formatting.

As a result:
- Table-style formatting is not visible to the featurizer
- Evaluation metrics may differ when Excel Tables are present

The evaluator supports:
- CNN-only mode
- Excel-object-only mode
- Hybrid aggregation mode

---

## Extensibility

Although the primary task is table detection, the architecture is intentionally general and can be extended to detect other spreadsheet patterns.

