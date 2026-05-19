"""
Compression-aware augmentation module for robust image deepfake detection.

Applies random chains of lossy distortions that simulate social-media
recompression artifacts:
  - JPEG compression (single and double pass)
  - WEBP compression
  - Gaussian blur
  - Downsampling / bicubic resize + upsample
  - Gaussian noise
  - Unsharp masking / sharpening
  - Color / saturation degradation
  - Posterization (screenshot banding)

All ops run on PIL Images so this slots before any torchvision transform.

Config format (JSON dict):
  Each key maps to {"prob": float, ...range params...}.
  Load via load_augment_config(path) or use DEFAULT_CONFIG directly.
"""

import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_CONFIG: Dict[str, Any] = {
    "jpeg":        {"prob": 0.60, "quality_min": 40,   "quality_max": 95},
    "double_jpeg": {"prob": 0.30, "q1_min": 65, "q1_max": 95, "q2_min": 40, "q2_max": 85},
    "webp":        {"prob": 0.20, "quality_min": 50,   "quality_max": 90},
    "blur":        {"prob": 0.30, "sigma_min": 0.3,    "sigma_max": 2.0},
    "resize":      {"prob": 0.35, "scale_min": 0.50,   "scale_max": 0.90},
    "noise":       {"prob": 0.25, "std_min": 0.005,    "std_max": 0.040},
    "sharpen":     {"prob": 0.25, "factor_min": 1.5,   "factor_max": 4.0},
    "color":       {"prob": 0.25, "brightness": 0.20,  "contrast": 0.20,
                    "saturation": 0.30, "hue": 0.05},
    "posterize":   {"prob": 0.10, "bits_min": 5,       "bits_max": 7},
}


def load_augment_config(path: Optional[str]) -> Dict[str, Any]:
    """Load augmentation config from JSON, falling back to DEFAULT_CONFIG."""
    if path and Path(path).exists():
        with open(path) as f:
            raw = json.load(f)
        # Strip comment keys and merge over defaults so new keys are addable
        cfg = dict(DEFAULT_CONFIG)
        for k, v in raw.items():
            if not k.startswith("_"):
                cfg[k] = v
        return cfg
    return dict(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Low-level helpers (PIL → PIL)
# ---------------------------------------------------------------------------

def _jpeg_encode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return Image.open(buf).copy().convert("RGB")


def _webp_encode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=4)
    buf.seek(0)
    return Image.open(buf).copy().convert("RGB")


def _add_gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0.0, std * 255.0, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resize_artifact(img: Image.Image, scale: float) -> Image.Image:
    """Downsample then restore to original size — introduces blocking/ringing."""
    w, h = img.size
    small_w, small_h = max(4, int(w * scale)), max(4, int(h * scale))
    img = img.resize((small_w, small_h), Image.BICUBIC)
    return img.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# CompressionAugment
# ---------------------------------------------------------------------------

class CompressionAugment:
    """
    Probabilistic compression-distortion augmentor.

    Args:
        config:     Augmentation config dict (see DEFAULT_CONFIG).
        prob_scale: Global multiplier for all probabilities (0–1 clamp applied).
                    Use < 1.0 to soften augmentations; > 1.0 to increase them.
    """

    _OPS_ORDER = [
        "resize",       # Downsample first (closest to real social-media flow)
        "jpeg",         # First JPEG pass
        "double_jpeg",  # Optional second JPEG pass (download → re-upload)
        "webp",         # Alternative: WEBP recompression
        "blur",         # Blur (after compression to simulate motion/focus artifacts)
        "sharpen",      # Sharpening (post-processing that reveals artifact halos)
        "noise",        # Additive noise
        "color",        # Color/saturation degradation
        "posterize",    # Screenshot banding
    ]

    def __init__(self, config: Dict[str, Any], prob_scale: float = 1.0):
        self.config = config
        self.prob_scale = float(prob_scale)

    def _prob(self, key: str) -> float:
        raw = self.config.get(key, {}).get("prob", 0.0)
        return min(1.0, max(0.0, raw * self.prob_scale))

    def _u(self, key: str, lo_field: str, hi_field: str) -> float:
        cfg = self.config[key]
        return random.uniform(cfg[lo_field], cfg[hi_field])

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random compression chain to a PIL Image; returns PIL Image."""
        img = img.convert("RGB")

        for op in self._OPS_ORDER:
            if random.random() >= self._prob(op):
                continue

            if op == "resize":
                scale = self._u("resize", "scale_min", "scale_max")
                img = _resize_artifact(img, scale)

            elif op == "jpeg":
                q = int(self._u("jpeg", "quality_min", "quality_max"))
                img = _jpeg_encode(img, q)

            elif op == "double_jpeg":
                q1 = int(self._u("double_jpeg", "q1_min", "q1_max"))
                q2 = int(self._u("double_jpeg", "q2_min", "q2_max"))
                img = _jpeg_encode(_jpeg_encode(img, q1), q2)

            elif op == "webp":
                q = int(self._u("webp", "quality_min", "quality_max"))
                img = _webp_encode(img, q)

            elif op == "blur":
                sigma = self._u("blur", "sigma_min", "sigma_max")
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

            elif op == "sharpen":
                factor = self._u("sharpen", "factor_min", "factor_max")
                img = ImageEnhance.Sharpness(img).enhance(factor)

            elif op == "noise":
                std = self._u("noise", "std_min", "std_max")
                img = _add_gaussian_noise(img, std)

            elif op == "color":
                cfg = self.config["color"]
                for Enh, field in [
                    (ImageEnhance.Brightness, "brightness"),
                    (ImageEnhance.Contrast,   "contrast"),
                    (ImageEnhance.Color,      "saturation"),
                ]:
                    delta = cfg[field]
                    factor = random.uniform(1.0 - delta, 1.0 + delta)
                    img = Enh(img).enhance(factor)

            elif op == "posterize":
                bits = int(self._u("posterize", "bits_min", "bits_max"))
                img = ImageOps.posterize(img, bits)

        return img

    def __repr__(self) -> str:
        active = [k for k in self._OPS_ORDER if self._prob(k) > 0]
        return f"CompressionAugment(ops={active}, prob_scale={self.prob_scale})"


# ---------------------------------------------------------------------------
# CompressionAwareTransform — wraps compression aug + base torchvision transform
# ---------------------------------------------------------------------------

class CompressionAwareTransform:
    """
    Composes CompressionAugment (PIL→PIL) with any base transform (PIL→Tensor).

    Use this to replace the dataset's .transform attribute after construction:

        dataset.transform = CompressionAwareTransform(
            CompressionAugment(cfg),
            build_transform(is_train=True, args=args),
        )
    """

    def __init__(self, compression_aug: CompressionAugment, base_transform):
        self.compression_aug = compression_aug
        self.base_transform = base_transform

    def __call__(self, img: Image.Image):
        img = self.compression_aug(img)       # PIL → PIL (random degradation)
        return self.base_transform(img)       # PIL → Tensor

    def __repr__(self) -> str:
        return (
            f"CompressionAwareTransform(\n"
            f"  aug={self.compression_aug},\n"
            f"  base={self.base_transform}\n"
            f")"
        )
