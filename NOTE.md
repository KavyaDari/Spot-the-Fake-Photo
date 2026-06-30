# Spot the Fake Photo – Note

**How I did it:** I extract 12 classical computer-vision features per image
— FFT/moiré frequency-domain peaks (a screen's pixel grid creates periodic
structure that real photos don't have), Laplacian sharpness, HSV colour
statistics (screens emit light rather than reflect it), noise-residual
variance, and edge-sharpness profile — then feed them into a logistic
regression classifier (`scikit-learn`). I tested 6 classifiers (Logistic
Regression, Random Forest, Gradient Boosting, SVM, k-NN, XGBoost) under
5-fold cross-validation; Logistic Regression tied for best and is by far the
smallest and fastest, so I kept it.

Getting useful accuracy took real iteration: my first dataset plateaued
around 80% no matter which classifier I used, which told me the problem was
data, not model. Diagnosing per-feature separability turned up a lighting
confound (my real photos were mostly daylight/outdoor, screen photos mostly
indoor/evening) and a labelling mistake (a few "screen" photos were actually
real-world railings whose repeating bars spuriously triggered the
frequency features). Fixing both — rebalancing lighting across both
classes and removing the mislabelled photos — brought accuracy from ~80% to
90.9%, with much lower fold-to-fold variance (±4.1% vs ±11.5%), meaning the
model generalises consistently now rather than getting lucky on certain
splits.

**Accuracy: 90.9% ± 4.1%**, 5-fold stratified cross-validation, on 111
self-collected photos (53 real, 58 screen). This is below the 95% target
and I'm reporting it honestly rather than rounding up. Remaining errors
cluster around genuinely hard cases I chose to keep rather than discard:
night photos with artificial lighting, real photos with glass/reflective
surfaces, and screen photos taken from far enough away that moiré is
naturally faint.

**Latency:** ~190ms per image, laptop CPU (measured directly).
**Cost:** $0 on-device; roughly $0.003–0.006 per 1,000 images on a cloud
server, assuming ~190ms CPU compute per call on something like AWS Lambda.

**What I'd improve with more time:** (1) More data, specifically targeting
the hard cases above rather than just more volume. (2) A properly
fine-tuned CNN — I tried transfer learning with a frozen MobileNetV3-Small
backbone, but it underperformed (58–73% per fold) on only ~110 images;
with 300+ images and some unfrozen backbone layers I'd expect it to beat
the classical approach. (3) Ensembling classical features with CNN-learned
features, since they likely capture complementary signal.
