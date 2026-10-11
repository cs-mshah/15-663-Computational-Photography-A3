import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from figures import grid
from fusion import denoise, divergence, fused_gradients, gradient, laplacian, poisson
from image_io import align_flash, exposure, read_image, save_image


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2) + "\n")


def solve(ambient, flash, sigma, tau, boundary, initial):
    field, coherence, saturation = fused_gradients(ambient, flash, sigma, tau)
    result, info = poisson(divergence(*field), boundary, initial, tolerance=1e-6)
    if not all(c["converged"] for c in info):
        raise RuntimeError("Poisson solver did not converge")
    return result, field, coherence, saturation, info


def part1(data, output, figures):
    ambient_path = data / "lamp/lamp_ambient.tif"
    flash_path = data / "lamp/lamp_flash.tif"
    a, f = [read_image(p, 1536) for p in (ambient_path, flash_path)]
    ratio = exposure(flash_path)["iso"] / exposure(ambient_path)["iso"]
    output.mkdir(parents=True, exist_ok=True)
    settings = [(2, 0.05), (8, 0.1), (32, 0.25)]
    methods = ["bilateral", "joint", "detail", "masked"]
    patches = []
    titles = []
    selected = None
    joint_small = None
    for spatial, intensity in settings:
        result = denoise(a, f, spatial, intensity, ratio)
        for method in methods:
            patches.append(result[method][240:600, 360:820])
            titles.append(method.capitalize())
        if (spatial, intensity) == (8, 0.1):
            selected = result
        if (spatial, intensity) == (2, 0.05):
            joint_small = result["joint"]
        print(f"Lamp filters {spatial}, {intensity} finished", flush=True)
    grid(
        figures / "lamp_parameters.jpg",
        patches,
        titles,
        columns=4,
        row_labels=["s=2, r=0.05", "s=8, r=0.10", "s=32, r=0.25"],
    )
    best = {**selected, "joint": joint_small}
    for method in methods:
        save_image(output / f"lamp_{method}.png", best[method])
    save_image(output / "lamp_artifact_mask.png", selected["mask"])
    grid(
        figures / "lamp_results.jpg",
        [a, f] + [best[m] for m in methods],
        [
            "Ambient",
            "Flash",
            "Bilateral",
            "Joint bilateral",
            "Detail transfer",
            "Masked result",
        ],
    )
    differences = [
        selected["joint"] - selected["bilateral"],
        selected["detail"] - selected["joint"],
        selected["masked"] - selected["detail"],
    ]
    grid(
        figures / "lamp_differences.png",
        differences,
        ["Joint - bilateral", "Detail - joint", "Masked - detail"],
        signed=True,
        limit=0.08,
    )
    grid(
        figures / "lamp_mask.jpg",
        [a, f, selected["mask"]],
        ["Ambient", "Flash", "Artifact mask"],
    )
    write_json(
        output / "settings.json",
        {
            "shape": list(a.shape),
            "tried": settings,
            "selected": {m: ([2, 0.05] if m == "joint" else [8, 0.1]) for m in methods},
            "exposure_ratio": ratio,
            "epsilon": 0.02,
            "shadow_threshold": 0.01,
            "specular_threshold": 0.95,
        },
    )


def part2(data, output, figures):
    paths = [data / f"museum/museum_{name}.png" for name in ("ambient", "flash")]
    a, f = [read_image(p, 452) for p in paths]
    output.mkdir(parents=True, exist_ok=True)
    div = divergence(*gradient(a))
    reconstruction_rows = []
    for tolerance, maximum in [(1e-6, 100), (1e-3, 4000), (1e-6, 4000)]:
        recovered, info = poisson(div, a, tolerance=tolerance, max_iterations=maximum)
        reconstruction_rows.append(
            {
                "tolerance": tolerance,
                "maximum": maximum,
                "iterations": [x["iterations"] for x in info],
                "residual": max(x["residual"] for x in info),
                "max_error": float(np.max(np.abs(recovered - a))),
                "rmse": float(np.sqrt(np.mean((recovered - a) ** 2))),
            }
        )
    operator_error = float(np.max(np.abs((div - laplacian(a))[1:-1, 1:-1])))
    parameter_images, parameter_titles = [], []
    for sigma in (10, 40, 80):
        for tau in (0.7, 0.9, 0.95):
            result, *_ = solve(a, f, sigma, tau, (a + f) / 2, (a + f) / 2)
            parameter_images.append(result)
            parameter_titles.append(f"sigma={sigma}, tau={tau}")
            print(f"Museum parameters {sigma}, {tau} finished", flush=True)
    grid(figures / "museum_parameters.jpg", parameter_images, parameter_titles)
    sigma, tau = 40, 0.7
    boundaries = {"ambient": a, "flash": f, "average": (a + f) / 2}
    boundary_images, boundary_info = [], {}
    for name, boundary in boundaries.items():
        result, _, _, _, info = solve(a, f, sigma, tau, boundary, (a + f) / 2)
        boundary_images.append(result)
        boundary_info[name] = [x["iterations"] for x in info]
    grid(
        figures / "museum_boundaries.jpg",
        boundary_images,
        ["Ambient boundary", "Flash boundary", "Average boundary"],
    )
    initializations = {"zero": np.zeros_like(a), **boundaries}
    initial_rows = []
    zero_result = None
    for name, initial in initializations.items():
        result, _, _, _, info = solve(a, f, sigma, tau, a, initial)
        if name == "zero":
            zero_result = result
        initial_rows.append(
            {
                "initial": name,
                "iterations": [x["iterations"] for x in info],
                "residual": max(x["residual"] for x in info),
                "difference_from_zero": float(np.max(np.abs(result - zero_result))),
            }
        )
    a, f = [read_image(p) for p in paths]
    result, field, coherence, saturation, info = solve(a, f, sigma, tau, a, a)
    save_image(output / "museum_fused.png", result)
    grid(figures / "museum_result.jpg", [a, f, result], ["Ambient", "Flash", "Fused"])
    fields = [gradient(a), gradient(f), field]
    grid(
        figures / "museum_gradients.png",
        [g[i] for i in (0, 1) for g in fields],
        ["Ambient x", "Flash x", "Fused x", "Ambient y", "Flash y", "Fused y"],
        signed=True,
        limit=0.08,
        colorbar_label="Derivative",
    )
    write_json(
        output / "metrics.json",
        {
            "study_shape": [319, 452],
            "operator_error": operator_error,
            "reconstruction": reconstruction_rows,
            "boundary_iterations": boundary_info,
            "initializations": initial_rows,
            "selected": {
                "sigma": sigma,
                "tau": tau,
                "boundary": "ambient",
                "initial": "ambient",
            },
            "final_shape": list(a.shape),
            "final_solver": info,
        },
    )
    print("Museum final result finished", flush=True)


def own_pair(data, name, crop, max_size):
    folder = data / "own" / name
    images = []
    for kind in ("ambient", "flash"):
        image = read_image(folder / f"{kind}.jpg")
        x0, y0, x1, y1 = crop
        image = image[y0:y1, x0:x1]
        scale = max_size / max(image.shape[:2])
        images.append(
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
    a, f = images
    f, alignment = align_flash(a, f)
    a, f = a[2:-2, 2:-2], f[2:-2, 2:-2]
    settings = {kind: exposure(folder / f"{kind}.jpg") for kind in ("ambient", "flash")}
    return (
        a,
        f,
        {
            "crop": crop,
            "alignment": alignment,
            "exposure": settings,
            "shape": list(a.shape),
        },
    )


def part3(data, output, figures):
    output.mkdir(parents=True, exist_ok=True)
    a, f, info = own_pair(data, "denoising", [1350, 1500, 5550, 4000], 1600)
    settings = info["exposure"]
    ratio = (
        settings["flash"]["iso"]
        * settings["flash"]["seconds"]
        / (settings["ambient"]["iso"] * settings["ambient"]["seconds"])
    )
    result = denoise(a, f, 6, 0.08, ratio)
    save_image(output / "desk_denoised.png", result["masked"])
    grid(
        figures / "desk_denoising.jpg",
        [a, f, result["masked"]],
        ["Ambient", "Flash", "Denoised"],
    )
    grid(
        figures / "desk_detail.jpg",
        [x[580:840, 550:830] for x in (a, f, result["masked"])],
        ["Ambient detail", "Flash detail", "Denoised detail"],
    )
    info.update({"sigma_s": 6, "sigma_r": 0.08, "exposure_ratio": ratio})
    write_json(output / "denoising_settings.json", info)
    print("Own denoising finished", flush=True)
    a, f, info = own_pair(data, "gradient", [480, 1550, 4200, 3800], 1000)
    result, _, _, _, solver_info = solve(a, f, 40, 0.9, a, a)
    save_image(output / "mug_fused.png", result)
    grid(figures / "mug_fusion.jpg", [a, f, result], ["Ambient", "Flash", "Fused"])
    info.update(
        {
            "sigma": 40,
            "tau": 0.9,
            "boundary": "ambient",
            "initial": "ambient",
            "solver": solver_info,
        }
    )
    write_json(output / "gradient_settings.json", info)
    print("Own gradient fusion finished", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--part", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, function in [("1", part1), ("2", part2), ("3", part3)]:
        if args.part in (name, "all"):
            function(args.data, args.output / f"part{name}", figures)


if __name__ == "__main__":
    main()
