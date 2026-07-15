# CGAN for Ground-Motion Response Spectra Prediction

**Author:** Pavan Kumar
**Affiliation:** Indian Institute of Technology - Madras

A Conditional Generative Adversarial Network (C-GAN) for predicting earthquake
response spectra (PGA + pseudo-spectral acceleration at 24 time periods),
conditioned on:

- Earthquake magnitude (`Mw`)
- Fault type
- Hypocentral depth
- Joyner-Boore distance (`R_JB`)
- `Vs30` (site condition)
- Direction

## Repository Structure

```
.
├── complete_model.py                  # Full CGAN training pipeline (generator + discriminator)
├── CGAN_under_deployment.py           # Load a trained generator and predict response spectra
├── plot_response_spectra_grid.py      # Restyled 4x4 grid plot: mean prediction vs recorded spectra
├── final_data.csv                     # Training/validation data (ground-motion records)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model
See `complete_model.py` for the full training pipeline (Generator/Discriminator
architecture, training loop, and residual diagnostics). Update the placeholder
paths (`<path to your file>`, model checkpoint paths, etc.) for your own data
and directory structure before running.

### 2. Run inference with a trained generator
```bash
python CGAN_under_deployment.py
```
This loads a trained generator checkpoint (`cgan_gen_1.hdf5`), predicts a
response spectrum for a chosen record, and plots the mean prediction with a
Monte-Carlo uncertainty band against the recorded spectrum.

### 3. Generate the multi-panel comparison grid
```bash
python plot_response_spectra_grid.py
```
Produces a 4x4 grid of Mean Prediction / Recorded Response Spectra / Standard
Deviation band plots across multiple records, styled for readability
(shared legend, colorblind-friendly palette, decluttered gridlines).

> **Note:** `plot_response_spectra_grid.py` currently demonstrates the
> visual style using a placeholder uncertainty band derived from the
> recorded spectra. Swap in your real `predictions.mean(axis=0)` /
> `predictions.std(axis=0)` from `make_predictions(...)` in
> `CGAN_under_deployment.py` to plot actual model output.

## License

Add a license of your choice (e.g., MIT) before publishing.
