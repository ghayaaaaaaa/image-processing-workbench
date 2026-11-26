"""Generate a local, reproducible image-processing walkthrough."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from .operations import (
    contact_sheet,
    create_demo_image,
    fourier_low_pass,
    grayscale,
    resize_by,
    rotate,
    translate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_demo(output_dir: Path, size: int = 320, seed: int = 17, cutoff: float = 0.12) -> Path:
    """Write demo PNGs and a hash manifest without replacing existing files."""
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError("output directory is not empty; choose a new path")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = create_demo_image(size=size, seed=seed)
    variants: dict[str, Image.Image] = {
        "original": original,
        "grayscale": grayscale(original),
        "resized 0.65x": resize_by(original, 0.65),
        "translated +24,+16": translate(original, 24, 16),
        "rotated 22 deg": rotate(original, 22),
        f"Fourier low-pass {cutoff:.2f}": fourier_low_pass(original, cutoff=cutoff),
    }

    artifacts: dict[str, str] = {}
    for index, (label, image) in enumerate(variants.items(), start=1):
        safe_label = (
            label.lower()
            .replace(" ", "-")
            .replace("+", "plus")
            .replace(".", "-")
        )
        filename = f"{index:02d}-{safe_label}.png"
        destination = output_dir / filename
        image.save(destination, format="PNG", optimize=True)
        artifacts[filename] = label

    sheet_name = "contact-sheet.png"
    contact_sheet(variants).save(output_dir / sheet_name, format="PNG", optimize=True)
    artifacts[sheet_name] = "Labeled comparison of the generated variants"

    manifest = {
        "project": "image-processing-workbench",
        "input": "Generated synthetic RGB pattern; no external or personal image is read.",
        "parameters": {"size": size, "seed": seed, "fourier_low_pass_cutoff": cutoff},
        "files": [
            {
                "name": filename,
                "operation": operation,
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": _sha256(output_dir / filename),
            }
            for filename, operation in artifacts.items()
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_dir
