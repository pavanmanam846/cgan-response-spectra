# Ground Motion Model for Acceleration Response Spectra using Conditional Generative Adversarial Network (CGAN)

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

</p>

<p align="center">
  <img src="images/project_overview.png" width="100%">
</p>

---

## Overview

This repository presents a **Conditional Generative Adversarial Network (CGAN)** framework for predicting **earthquake acceleration response spectra** using earthquake source and site-specific parameters.

Traditional stochastic finite-source simulations are computationally expensive for generating large numbers of response spectra. The proposed CGAN learns the underlying statistical distribution of recorded ground motions and efficiently generates realistic response spectra conditioned on earthquake characteristics.

The framework provides a fast and flexible surrogate model suitable for seismic hazard assessment, structural engineering applications, and uncertainty quantification.

---

## Key Features

- Deep learning-based prediction of earthquake response spectra
- Conditional Generative Adversarial Network (CGAN) architecture
- Seven earthquake and site-specific conditioning variables
- Monte Carlo sampling for uncertainty estimation
- Comparison of predicted and recorded response spectra
- PyTorch implementation for training and deployment

---

## Conditional GAN Architecture

The proposed framework employs a **Conditional Generative Adversarial Network (CGAN)** consisting of two competing neural networks trained simultaneously through adversarial learning.

<p align="center">
  <img src="images/cgan_workflow.png" width="90%">
</p>


### Generator

The Generator receives:

- Random Gaussian noise
- Earthquake conditional parameters

and generates synthetic acceleration response spectra.

Its objective is to learn the underlying distribution of recorded spectra while producing realistic synthetic samples.

### Discriminator

The Discriminator receives:

- Recorded response spectra
- Generated response spectra
- Corresponding conditional variables

and predicts whether the supplied spectra are **real** or **synthetic**.

During training, the Generator continuously improves by attempting to fool the Discriminator, while the Discriminator learns to distinguish generated spectra from recorded data. The adversarial training converges when the generated spectra become statistically indistinguishable from real observations.

---

## Conditional Input Variables

The CGAN conditions the generated response spectra using the following earthquake and site parameters.

| Variable | Description |
|-----------|-------------|
| Mw | Moment Magnitude |
| RJB | Joyner-Boore Distance |
| log(RJB) | Logarithmic Joyner-Boore Distance |
| Hdepth | Hypocentral Depth |
| log(VS30) | Site Shear Wave Velocity |
| Fault Type | Earthquake Fault Mechanism |
| Direction | Ground Motion Component |

---

## Repository Structure

```text
.
├── complete_model.py
├── CGAN_under_deployment.py
├── plot_response_spectra_grid.py
├── final_data.csv
├── requirements.txt
├── README.md
└── images
    ├── project_overview.png
    ├── cgan_workflow.png
    └── response_spectra_grid.png
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/pavanmanam846/cgan-response-spectra.git

cd cgan-response-spectra
```

Install the required packages

```bash
pip install -r requirements.txt
```

Alternatively,

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn torch torchvision
```

---

## Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.stats import kurtosis
from scipy.stats import skew

from sklearn.metrics import r2_score

import math
import csv
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
```

---

## Model Training

Run the training pipeline

```bash
python complete_model.py
```

This script:

- Loads the earthquake response spectra dataset
- Performs preprocessing and normalization
- Builds the Generator and Discriminator networks
- Trains the Conditional GAN
- Saves the trained model weights
- Plots Generator and Discriminator training losses

For faster convergence and improved training performance, GPU acceleration is recommended.

---

## Model Deployment

Once the Generator has been trained,

```bash
python CGAN_under_deployment.py
```

This deployment script

- Loads the trained Generator
- Accepts earthquake conditional parameters
- Generates multiple stochastic response spectra
- Computes the mean prediction
- Computes prediction uncertainty
- Compares predicted and recorded spectra

---

## Model Validation

The prediction capability of the trained CGAN is evaluated by comparing generated response spectra with recorded earthquake observations.

<p align="center">
  <img src="images/response_spectra_grid.png" width="100%">
</p>

The comparison plots present:

- Recorded Response Spectra
- Mean CGAN Prediction
- Prediction Standard Deviation

The close agreement between predicted and recorded spectra demonstrates the capability of the proposed model to reproduce both the spectral characteristics and variability of earthquake ground motions.

---

## Applications

This framework can be applied to:

- Ground Motion Simulation
- Earthquake Engineering
- Structural Dynamics
- Performance-Based Earthquake Engineering (PBEE)
- Probabilistic Seismic Hazard Assessment (PSHA)
- Structural Reliability Analysis
- Seismic Risk Assessment

---

## Future Work

Potential extensions include:

- Physics-informed Conditional GANs
- Multi-component ground motion prediction
- Bayesian uncertainty quantification
- Transfer learning for regional seismic datasets
- Real-time response spectrum generation

---

## Citation

If you use this repository in your research, please cite the associated publication.

```text
Pavan Kumar Manam

Ground Motion Model for Acceleration Response Spectra
using Conditional Generative Adversarial Network (CGAN)

```

---

## Author

**Pavan Kumar Manam**


📧 **manamkeerthipavankumar@gmail.com**

---

## License

This repository is released for **academic and research purposes**.

If you use this work in your research, please cite the corresponding publication.
