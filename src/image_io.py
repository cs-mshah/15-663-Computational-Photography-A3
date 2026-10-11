from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image(path, max_size=0):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = image[..., :3][..., ::-1].astype(np.float64) / np.iinfo(image.dtype).max
    if max_size and max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def exposure(path):
    with Image.open(path) as image:
        exif = image.getexif().get_ifd(34665)
    return {
        "iso": int(exif[34855]),
        "seconds": float(exif.get(33434, 1)),
        "aperture": float(exif.get(33437, 1)),
    }


def align_flash(ambient, flash):
    cv2.setRNGSeed(0)
    gray = [np.uint8(np.clip(x.mean(axis=2), 0, 1) * 255) for x in (ambient, flash)]
    sift = cv2.SIFT_create(nfeatures=4000)
    (ka, da), (kf, df) = [sift.detectAndCompute(x, None) for x in gray]
    matches = cv2.BFMatcher().knnMatch(df, da, k=2)
    matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    source = np.float32([kf[m.queryIdx].pt for m in matches])
    target = np.float32([ka[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=1
    )
    if matrix is None or inliers.sum() < 10:
        raise ValueError("Too few consistent matches to align the pair")
    height, width = ambient.shape[:2]
    aligned = cv2.warpAffine(
        flash,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    errors = target - (source @ matrix[:, :2].T + matrix[:, 2])
    info = {
        "matrix": matrix.tolist(),
        "inliers": int(inliers.sum()),
        "median_error": float(
            np.median(np.linalg.norm(errors[inliers.ravel() > 0], axis=1))
        ),
    }
    return aligned, info
