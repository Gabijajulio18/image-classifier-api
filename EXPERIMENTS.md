# Experiment Tracking

This file tracks training experiments for the flower classification project. Each run records the changes applied,
metrics on the validation/test set and any useful note.

| Version | Description | Val Acc | Test acc | Notes |
|---------|-------------|--------|----------|-------|
| v1 | Baseline CNN | - | 0.689 | Initial model with three Conv2D blocks and Adam optimizer. | 
| v2 | Dropout 0.5, BatchNorm, Aug(0.1), Adam | 0.565 | 0.602 | Over-regularized? Needs tuning |
| v3 | Dropout 0.3, BatchNorm, Aug(0.1), Adam | 0.67 | 0.703 |
| v4 | No Dropout , BatchNorm, Aug(0.1), Adam(1e-4) | 0.642 | 0.648 |After tuning  |
| v5 | MobileNetV2 transfer learning | 0.904 | 0.906 | Pretrained base model fine-tuned for 10 epochs |