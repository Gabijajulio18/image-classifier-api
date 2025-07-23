# Experiment Tracking 

This file tracks training experiments for the flower classification project. Each run records the changes applied,
metrics on the validation/test set and any useful note.

| Version | Description | Va Acc | Test acc | Notes |
|---------|-------------|--------|----------|-------|
| v1 | Baseline CNN | - | 0.689 | Initial model with three Conv2D blocks and Adam optimizer. | 
| v2 | Dropout 0.5, BatchNorm, Aug(0.1), Adam | 0.565 | 0.602 | Over-regularized? Needs tuning |
| v3 | Dropout 0.3, BatchNorm, Aug(0.1), Adam | 0.67 | 0.703 |
| v4 | Hyperparameter Tuning