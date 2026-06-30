#!/usr/bin/env python3
"""
predict.py  –  Spot the Fake Photo
Usage:  python predict.py <image_path>
Output: a float in [0, 1]   (0 = real photo,  1 = photo of a screen)

Approach: hybrid classical-CV feature extraction + logistic regression.
No GPU, no heavy model weights.  ~5-15 ms on a laptop CPU.
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# ── HEIC support (iPhone photos) ──
try:
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_SUPPORT = True
except ImportError:
    _HEIC_SUPPORT = False


def _read_image_any_format(image_path: str) -> np.ndarray:
    """
    Read an image file into a BGR numpy array (OpenCV convention),
    supporting HEIC/HEIF (iPhone) in addition to formats cv2 handles natively.
    """
    ext = Path(image_path).suffix.lower()
    if ext in (".heic", ".heif"):
        if not _HEIC_SUPPORT:
            raise ValueError(
                f"Cannot read {image_path}: HEIC support not installed. "
                f"Run: pip install pillow-heif"
            )
        pil_img = Image.open(image_path).convert("RGB")
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    bgr = cv2.imread(image_path)
    if bgr is None:
        # Fallback: try PIL for anything cv2 chokes on (rare formats, odd encodings)
        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Cannot read image: {image_path}")
    return bgr

# ──────────────────────────────────────────────
# 1.  FEATURE EXTRACTION  (the core of the detector)
# ──────────────────────────────────────────────

def fft_peak_ratio(gray: np.ndarray) -> float:
    """
    Screen photos have periodic pixel-grid / subpixel structure that
    shows up as sharp peaks in the 2-D power spectrum.
    We measure how much energy is concentrated in the top-N peaks
    relative to the total – screens score high, real photos score low.
    """
    # Work on a 512×512 crop from the centre (fast + consistent)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    half = 256
    crop = gray[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    crop = cv2.resize(crop, (512, 512))

    f = np.fft.fft2(crop.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    # Zero out the DC component (centre)
    mag[254:258, 254:258] = 0

    total = mag.sum() + 1e-9
    # Top-200 pixel values as fraction of total energy
    flat = mag.flatten()
    flat.sort()
    top200 = flat[-200:].sum()
    return float(top200 / total)


def moire_score(gray: np.ndarray) -> float:
    """
    Moiré / regular grid patterns → high energy at mid frequencies.
    Compute ratio of mid-band energy to total (excluding DC).
    """
    f = np.fft.fft2(cv2.resize(gray, (256, 256)).astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2

    # Rings
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    inner, outer = 20, 80          # mid-frequency band
    mid_mask = (dist >= inner) & (dist <= outer)
    dc_mask  = dist < inner

    mid_energy = mag[mid_mask].sum()
    dc_energy  = mag[dc_mask].sum()
    total      = mag.sum() + 1e-9

    # Exclude DC for ratio
    return float(mid_energy / (total - dc_energy + 1e-9))


def laplacian_stats(gray: np.ndarray):
    """
    Sharpness via Laplacian.  Screen photos often have unnaturally sharp
    or flat regions (the display pixels are perfectly sharp) compared to
    natural optical blur in real photos.
    Returns (variance, kurtosis-proxy).
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mean, std = cv2.meanStdDev(lap)
    mean, std = float(mean[0,0]), float(std[0,0]) + 1e-9
    var = std ** 2
    # Excess kurtosis proxy: how 'peaky' is the distribution?
    # (manual z*z*z*z is much faster than z**4 in numpy)
    z = (lap - mean) / std
    z2 = z * z
    kurt = float(np.mean(z2 * z2))
    return var, kurt


def color_stats(bgr: np.ndarray):
    """
    Screens emit light; real scenes reflect it.
    Screens tend to have:
      - higher mean saturation in HSV
      - lower std of V channel (more uniform luminance across pixels)
      - slightly different channel correlation structure
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    mean_s  = float(np.mean(s_ch))
    std_s   = float(np.std(s_ch))
    mean_v  = float(np.mean(v_ch))
    std_v   = float(np.std(v_ch))

    # Channel correlation: B-G, G-R  (screens often compress/alter these)
    b = bgr[:,:,0].astype(np.float32).flatten()
    g = bgr[:,:,1].astype(np.float32).flatten()
    r = bgr[:,:,2].astype(np.float32).flatten()
    corr_bg = float(np.corrcoef(b, g)[0, 1])
    corr_gr = float(np.corrcoef(g, r)[0, 1])

    return mean_s, std_s, mean_v, std_v, corr_bg, corr_gr


def noise_analysis(gray: np.ndarray):
    """
    Camera sensor noise is random; screens introduce structured noise.
    We subtract a Gaussian-smoothed version and analyse the residual.
    """
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - smooth.astype(np.float32)
    mean, std = cv2.meanStdDev(residual)
    mean, std = float(mean[0,0]), float(std[0,0]) + 1e-9
    var  = std ** 2
    z = (residual - mean) / std
    skew = float(np.mean(z * z * z))
    return var, skew


def spectral_peakiness(gray: np.ndarray) -> float:
    """
    Better moiré/grid detector: looks for a small number of NARROW, SHARP
    peaks in the frequency spectrum (characteristic of a periodic pixel
    grid), as opposed to just 'high energy somewhere' which is too crude
    and picks up texture from real photos too.

    We compute the ratio of the single strongest non-DC frequency bin
    to the local neighborhood average — a true periodic source will have
    one bin far above its neighbors; natural image texture will not.
    """
    crop = cv2.resize(gray, (256, 256)).astype(np.float32)
    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2

    # mask out DC region
    mag_masked = mag.copy()
    mag_masked[cy-10:cy+10, cx-10:cx+10] = 0

    # find peak location
    peak_idx = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)
    peak_val = mag_masked[peak_idx]

    # local neighborhood average around the peak (excluding peak itself)
    py, px = peak_idx
    y0, y1 = max(0, py-15), min(rows, py+15)
    x0, x1 = max(0, px-15), min(cols, px+15)
    neighborhood = mag_masked[y0:y1, x0:x1].copy()
    neighborhood[py-y0, px-x0] = 0  # exclude peak
    local_avg = neighborhood.mean() + 1e-9

    return float(peak_val / local_avg)


def edge_sharpness_profile(gray: np.ndarray) -> float:
    """
    Real-world optical blur causes edges to transition gradually.
    Screen-captured images (re-photographed) often have edges that are
    either unnaturally crisp (if screen is in sharp focus) OR have
    a secondary blur signature from double-lens capture.
    We measure the average edge transition width via gradient magnitude
    concentration.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Only look at strong edges
    threshold = np.percentile(grad_mag, 90)
    strong_edges = grad_mag[grad_mag > threshold]
    if len(strong_edges) == 0:
        return 0.0
    return float(np.std(strong_edges) / (np.mean(strong_edges) + 1e-9))


def local_contrast_uniformity(gray: np.ndarray) -> float:
    """
    Screens have very uniform local contrast within sub-regions
    (driven by display panel grid); real photos have organic variation.
    Measure variance of local std-dev across a grid of patches.
    """
    h, w = gray.shape
    patch = 32
    stds = []
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            block = gray[y:y+patch, x:x+patch]
            stds.append(np.std(block))
    if len(stds) < 2:
        return 0.0
    return float(np.std(stds) / (np.mean(stds) + 1e-9))


def extract_features(image_path: str) -> np.ndarray:
    """
    Load image, extract all features, return 1-D numpy array.
    """
    bgr = _read_image_any_format(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize for consistency / speed (keep aspect)
    max_side = 1024
    h, w = bgr.shape[:2]
    scale = min(max_side / h, max_side / w, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    f1   = fft_peak_ratio(gray)
    f2   = moire_score(gray)
    f3, _ = laplacian_stats(gray)             # keep var, drop noisy kurtosis
    f5, f6, f7, f8, f9, f10 = color_stats(bgr)
    f11, _ = noise_analysis(gray)             # keep var, drop noisy skew
    f13  = spectral_peakiness(gray)           # sharper peak detector
    f14  = edge_sharpness_profile(gray)
    # local_contrast_uniformity dropped: measured d=0.19 on real data (noise)

    return np.array([f1, f2, f3, f5, f6, f7, f8, f9, f10, f11, f13, f14],
                    dtype=np.float32)


# ──────────────────────────────────────────────
# 2.  CLASSIFIER
#     We ship a logistic regression with hand-calibrated weights.
#     If you have labelled data, run train.py to learn better weights
#     and they will be saved to model_weights.npz and auto-loaded here.
# ──────────────────────────────────────────────

# ── default weights (calibrated by analysis, no training data needed) ──
# Feature order: fft_peak, moire, lap_var,
#                mean_s, std_s, mean_v, std_v, corr_bg, corr_gr,
#                noise_var, spectral_peakiness, edge_sharpness
_DEFAULT_WEIGHTS = np.array([
     4.0,   # fft_peak_ratio
     2.0,   # moire_score
    -0.8,   # laplacian_var
     1.2,   # mean_saturation
    -0.5,   # std_saturation
    -1.0,   # mean_value
    -0.8,   # std_value
    -1.0,   # corr_bg
    -1.0,   # corr_gr
    -1.5,   # noise_var
     3.0,   # spectral_peakiness
     1.0,   # edge_sharpness_profile
], dtype=np.float32)
_DEFAULT_BIAS = np.float32(-4.0)

# Feature normalisation constants (μ, σ) — rough defaults, overwritten by train.py
_MU    = np.array([0.06, 0.49, 1800.0, 60.0, 42.0, 105.0, 61.0, 0.96, 0.97, 110.0, 5.0, 0.5], dtype=np.float32)
_SIGMA = np.array([0.04, 0.06, 2800.0, 25.0, 14.0,  33.0, 17.0, 0.05, 0.06,  150.0, 5.0, 0.3], dtype=np.float32)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def load_model():
    """Load learned weights if available, else use defaults."""
    weights_path = Path(__file__).parent / "model_weights.npz"
    if weights_path.exists():
        data = np.load(weights_path)
        return data["weights"], data["bias"], data["mu"], data["sigma"]
    return _DEFAULT_WEIGHTS, _DEFAULT_BIAS, _MU, _SIGMA


def predict(image_path: str) -> float:
    """
    Main predictor.
    Returns float in [0, 1]:  0 = real photo,  1 = screen / recapture.
    """
    weights, bias, mu, sigma = load_model()
    feats = extract_features(image_path)
    feats_norm = (feats - mu) / (sigma + 1e-9)
    score = _sigmoid(float(np.dot(weights, feats_norm) + bias))
    return score


# ──────────────────────────────────────────────
# 3.  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    score = predict(img_path)
    ms = (time.perf_counter() - t0) * 1000

    print(f"{score:.4f}")
    print(f"# {'SCREEN' if score >= 0.5 else 'REAL'}  |  latency: {ms:.1f} ms", file=sys.stderr)