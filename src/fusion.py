import cv2
import numpy as np


def bilateral(image, guide, sigma_s, sigma_r):
    if image.shape != guide.shape or sigma_s <= 0 or sigma_r <= 0:
        raise ValueError("Matching image shapes and positive sigmas are required")
    image = np.asarray(image, dtype=np.float64)
    guide = np.asarray(guide, dtype=np.float64)
    margin = min(0.01, sigma_r / 2)
    low, high = guide.min() - margin, guide.max() + margin
    segments = int(np.ceil((high - low) / sigma_r))
    step = (high - low) / segments
    result = np.zeros_like(image)
    for center in np.linspace(low, high, segments + 1):
        weights = np.exp(-0.5 * ((guide - center) / sigma_r) ** 2)
        numerator = cv2.GaussianBlur(weights * image, (0, 0), sigma_s)
        denominator = cv2.GaussianBlur(weights, (0, 0), sigma_s)
        filtered = numerator / np.maximum(denominator, 1e-300)
        interpolation = np.maximum(1 - np.abs(guide - center) / step, 0)
        result += interpolation * filtered
    return result


def linearize(image):
    return np.where(image <= 0.0404482, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def artifact_mask(
    ambient,
    flash,
    exposure_ratio,
    shadow_threshold=0.01,
    specular_threshold=0.95,
    radius=2,
):
    luminance = np.array([0.2126, 0.7152, 0.0722])
    a = linearize(ambient) @ luminance * exposure_ratio
    f = linearize(flash) @ luminance
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    masks = []
    for mask in (f - a <= shadow_threshold, f >= specular_threshold):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks.append(cv2.dilate(mask, kernel))
    mask = np.maximum(*masks).astype(np.float64)
    return cv2.GaussianBlur(mask, (0, 0), max(radius, 1))


def denoise(
    ambient,
    flash,
    sigma_s=8,
    sigma_r=0.1,
    exposure_ratio=1,
    epsilon=0.02,
    shadow_threshold=0.01,
):
    base = bilateral(ambient, ambient, sigma_s, sigma_r)
    joint = bilateral(ambient, flash, sigma_s, sigma_r)
    flash_base = bilateral(flash, flash, sigma_s, sigma_r)
    detail = joint * (flash + epsilon) / (flash_base + epsilon)
    mask = artifact_mask(ambient, flash, exposure_ratio, shadow_threshold)
    final = (1 - mask[..., None]) * detail + mask[..., None] * base
    return {
        "bilateral": base,
        "joint": joint,
        "detail": detail,
        "masked": final,
        "mask": mask,
    }


def gradient(image):
    dx, dy = np.zeros_like(image), np.zeros_like(image)
    dx[:, :-1] = np.diff(image, axis=1)
    dy[:-1] = np.diff(image, axis=0)
    return dx, dy


def divergence(dx, dy):
    result = dx + dy
    result[:, 1:] -= dx[:, :-1]
    result[1:] -= dy[:-1]
    return result


def laplacian(image):
    result = -4 * image
    result[1:] += image[:-1]
    result[:-1] += image[1:]
    result[:, 1:] += image[:, :-1]
    result[:, :-1] += image[:, 1:]
    return result


def poisson(div, boundary, initial=None, tolerance=1e-6, max_iterations=4000):
    if div.ndim == 3:
        channels = [
            poisson(
                div[..., c],
                boundary[..., c],
                None if initial is None else initial[..., c],
                tolerance,
                max_iterations,
            )
            for c in range(div.shape[2])
        ]
        return np.stack([x[0] for x in channels], axis=2), [x[1] for x in channels]
    image = (
        np.zeros_like(div, dtype=np.float64)
        if initial is None
        else initial.astype(np.float64).copy()
    )
    image[[0, -1], :] = boundary[[0, -1], :]
    image[:, [0, -1]] = boundary[:, [0, -1]]
    interior = np.zeros_like(image)
    interior[1:-1, 1:-1] = 1
    residual = interior * (div - laplacian(image))
    direction = residual.copy()
    delta = np.sum(residual * residual)
    iteration = 0
    while delta > tolerance**2 and iteration < max_iterations:
        q = laplacian(direction)
        alpha = delta / np.sum(direction * q)
        image += alpha * direction
        residual = interior * (residual - alpha * q)
        new_delta = np.sum(residual * residual)
        direction = residual + (new_delta / delta) * direction
        delta = new_delta
        iteration += 1
    norm = float(np.linalg.norm((div - laplacian(image))[1:-1, 1:-1]))
    return image, {
        "iterations": iteration,
        "residual": norm,
        "converged": norm <= tolerance * 1.01,
    }


def fused_gradients(ambient, flash, sigma=40, tau=0.9):
    ax, ay = gradient(ambient)
    fx, fy = gradient(flash)
    denominator = np.hypot(ax, ay) * np.hypot(fx, fy)
    coherence = np.divide(
        np.abs(ax * fx + ay * fy),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    coherence = np.clip(coherence, 0, 1)
    saturation = (np.tanh(sigma * (flash - tau)) + 1) / 2
    weight = (1 - saturation) * coherence
    gx = weight * fx + (1 - weight) * ax
    gy = weight * fy + (1 - weight) * ay
    return (gx, gy), coherence, saturation
