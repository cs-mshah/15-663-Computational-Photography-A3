import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def grid(
    path,
    images,
    titles,
    columns=3,
    row_labels=None,
    signed=False,
    limit=0.1,
    colorbar_label="Signed difference",
):
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns, 2.35 * rows),
        squeeze=False,
        layout="constrained",
    )
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i >= len(images):
            ax.set_visible(False)
            continue
        image = images[i]
        if signed:
            if image.ndim == 3:
                image = image.mean(axis=2)
            artist = ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(titles[i], fontsize=11)
        if row_labels and i % columns == 0:
            ax.set_ylabel(row_labels[i // columns], fontsize=11)
    if signed:
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.7, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)
