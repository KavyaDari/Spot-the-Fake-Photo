# Spot the Fake Photo

Detects whether an image is a real photo or a photo of a screen
(a "recapture") — built for the SalesCode AI take-home assignment.

## Quick start

```bash
pip install -r requirements.txt
python predict.py path/to/image.jpg
```

Outputs a float from 0 (real) to 1 (screen) on stdout, and latency info on
stderr.

## Files

- `predict.py` — the one-line predictor (required deliverable)
- `train.py` — trains the model on `real/` and `screen/` folders, saves
  `model_weights.npz`
- `model_weights.npz` — trained logistic regression weights (loaded
  automatically by `predict.py`)
- `NOTE.md` — write-up: approach, honest accuracy, latency/cost, and what
  I'd improve

## Approach

Classical CV feature extraction (FFT/moiré analysis, Laplacian sharpness,
colour statistics, noise residuals) feeding a logistic regression
classifier. See `NOTE.md` for accuracy numbers and what I'd improve with
more time.

## Re-training on your own data

```bash
python train.py --real real/ --screen screen/
```

Expects two folders of images (jpg/png/heic). Reports 5-fold
cross-validated accuracy and saves weights to `model_weights.npz`, which
`predict.py` auto-loads.
