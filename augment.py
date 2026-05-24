#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _gaussian_blur(img: Image.Image, severity: str) -> Image.Image:
    if severity == "light":
        radius = random.uniform(0.3, 1.0)
    elif severity == "medium":
        radius = random.uniform(0.5, 2.0)
    else:
        radius = random.uniform(1.0, 3.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _jpeg_compression(img: Image.Image, severity: str) -> Image.Image:
    import io
    if severity == "light":
        quality = random.randint(70, 85)
    elif severity == "medium":
        quality = random.randint(55, 75)
    else:
        quality = random.randint(40, 60)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _brightness_contrast(img: Image.Image, severity: str) -> Image.Image:
    if severity == "light":
        b_range = (0.85, 1.15)
        c_range = (0.85, 1.15)
    elif severity == "medium":
        b_range = (0.75, 1.25)
        c_range = (0.75, 1.25)
    else:
        b_range = (0.60, 1.40)
        c_range = (0.60, 1.40)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*b_range))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*c_range))
    return img


def _rotation(img: Image.Image, severity: str) -> Image.Image:
    if severity == "light":
        angle = random.uniform(-1.5, 1.5)
    elif severity == "medium":
        angle = random.uniform(-2.5, 2.5)
    else:
        angle = random.uniform(-3.0, 3.0)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def _perspective_warp(img: Image.Image, severity: str) -> Image.Image:
    w, h = img.size
    if severity == "light":
        max_shift = 0.02
    elif severity == "medium":
        max_shift = 0.035
    else:
        max_shift = 0.05

    def jitter():
        return random.uniform(-max_shift, max_shift)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (jitter() * w,       jitter() * h),
        (w + jitter() * w,   jitter() * h),
        (w + jitter() * w,   h + jitter() * h),
        (jitter() * w,       h + jitter() * h),
    ]

    matrix = []
    for (x1, y1), (x2, y2) in zip(src, dst):
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])

    B = np.array([coord for pt in dst for coord in pt], dtype=float)
    A = np.array(matrix, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs.tolist(), resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def _salt_pepper_noise(img: Image.Image, severity: str) -> Image.Image:
    if severity == "light":
        density = random.uniform(0.005, 0.010)
    elif severity == "medium":
        density = random.uniform(0.008, 0.015)
    else:
        density = random.uniform(0.012, 0.020)

    arr = np.array(img, dtype=np.uint8)
    total = arr.shape[0] * arr.shape[1]
    n_salt = int(total * density / 2)
    n_pepper = int(total * density / 2)

    coords_salt = [np.random.randint(0, arr.shape[0], n_salt), np.random.randint(0, arr.shape[1], n_salt)]
    coords_pepper = [np.random.randint(0, arr.shape[0], n_pepper), np.random.randint(0, arr.shape[1], n_pepper)]

    arr[coords_salt[0], coords_salt[1]] = 255
    arr[coords_pepper[0], coords_pepper[1]] = 0
    return Image.fromarray(arr)


def _horizontal_shear(img: Image.Image, severity: str) -> Image.Image:
    if severity == "light":
        shear_deg = random.uniform(-2, 2)
    elif severity == "medium":
        shear_deg = random.uniform(-3.5, 3.5)
    else:
        shear_deg = random.uniform(-5, 5)

    shear_rad = math.tan(math.radians(shear_deg))
    w, h = img.size
    coeffs = (1, shear_rad, -shear_rad * h / 2, 0, 1, 0)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def _crop_resize(img: Image.Image, severity: str) -> Image.Image:
    w, h = img.size
    if severity == "light":
        frac = random.uniform(0.03, 0.05)
    elif severity == "medium":
        frac = random.uniform(0.04, 0.06)
    else:
        frac = random.uniform(0.05, 0.08)

    left = int(w * random.uniform(0, frac))
    upper = int(h * random.uniform(0, frac))
    right = int(w * (1 - random.uniform(0, frac)))
    lower = int(h * (1 - random.uniform(0, frac)))
    right = max(right, left + 1)
    lower = max(lower, upper + 1)
    return img.crop((left, upper, right, lower)).resize((w, h), Image.BICUBIC)


def _to_grayscale_rgb(img: Image.Image, severity: str) -> Image.Image:
    return img.convert("L").convert("RGB")


def _dilation_erosion(img: Image.Image, severity: str) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)
    is_dark_text = arr.mean() > 128

    if severity == "heavy":
        kernel_size = 2
    else:
        kernel_size = 1

    from PIL import ImageFilter
    if is_dark_text:
        filt = ImageFilter.MinFilter(size=kernel_size * 2 + 1)
    else:
        filt = ImageFilter.MaxFilter(size=kernel_size * 2 + 1)
    return img.filter(filt)


_TRANSFORMS = [
    _gaussian_blur,
    _jpeg_compression,
    _brightness_contrast,
    _rotation,
    _perspective_warp,
    _salt_pepper_noise,
    _horizontal_shear,
    _crop_resize,
    _to_grayscale_rgb,
    _dilation_erosion,
]

_SEVERITY_N_TRANSFORMS = {
    "light":  (1, 1),
    "medium": (2, 3),
    "heavy":  (3, 5),
}


def augment_image(image: Image.Image, severity: str = "medium") -> Image.Image:
    if severity == "none":
        return image.convert("RGB") if image.mode != "RGB" else image

    if severity not in _SEVERITY_N_TRANSFORMS:
        raise ValueError(f"severity must be one of: none, light, medium, heavy. Got: {severity!r}")

    img = image.convert("RGB")
    lo, hi = _SEVERITY_N_TRANSFORMS[severity]
    n = random.randint(lo, hi)
    chosen = random.sample(_TRANSFORMS, n)

    grayscale_fn = _to_grayscale_rgb
    if grayscale_fn in chosen:
        chosen.remove(grayscale_fn)
        if random.random() < 0.40:
            img = grayscale_fn(img, severity)
    else:
        if random.random() < 0.40:
            img = grayscale_fn(img, severity)

    for fn in chosen:
        try:
            img = fn(img, severity)
        except Exception:
            pass

    return img.convert("RGB")
