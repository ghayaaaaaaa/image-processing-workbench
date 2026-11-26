"""Small image operations used by the synthetic CLI demonstration."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


RGB = tuple[int, int, int]


def create_demo_image(size: int = 320, seed: int = 17) -> Image.Image:
    """Create a reproducible image from shapes rather than external files."""
    if not 128 <= size <= 1024:
        raise ValueError("size must be between 128 and 1024 pixels")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), (16, 25, 38))
    draw = ImageDraw.Draw(image)
    margin = size // 12
    step = max(12, size // 16)

    for coordinate in range(margin, size - margin, step):
        draw.line((coordinate, margin, coordinate, size - margin), fill=(30, 45, 62), width=1)
        draw.line((margin, coordinate, size - margin, coordinate), fill=(30, 45, 62), width=1)

    center = size // 2
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=max(8, size // 24),
        outline=(87, 121, 161),
        width=max(2, size // 128),
    )
    draw.ellipse(
        (center - size // 5, center - size // 5, center + size // 5, center + size // 5),
        fill=(40, 90, 130),
        outline=(123, 196, 226),
        width=max(2, size // 96),
    )

    points: list[tuple[int, int]] = []
    for index in range(24):
        x = margin + int(index * (size - 2 * margin) / 23)
        phase = index / 23 * math.tau
        y = center + int(math.sin(phase) * size * 0.22)
        points.append((x, y))
    draw.line(points, fill=(255, 190, 90), width=max(2, size // 72), joint="curve")

    for _ in range(16):
        x = int(rng.integers(margin, size - margin))
        y = int(rng.integers(margin, size - margin))
        radius = int(rng.integers(max(2, size // 96), max(4, size // 48)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(107, 215, 181))
    return image


def grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def resize_by(image: Image.Image, scale: float) -> Image.Image:
    if not math.isfinite(scale) or not 0.25 <= scale <= 4.0:
        raise ValueError("scale must be between 0.25 and 4.0")
    width, height = image.size
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(target, resample=Image.Resampling.LANCZOS)


def translate(image: Image.Image, dx: int, dy: int, fill: RGB = (16, 25, 38)) -> Image.Image:
    source = image.convert("RGB")
    output = Image.new("RGB", source.size, fill)
    output.paste(source, (dx, dy))
    return output


def rotate(image: Image.Image, angle: float) -> Image.Image:
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return image.convert("RGB").rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(16, 25, 38),
    )


def fourier_low_pass(image: Image.Image, cutoff: float = 0.12) -> Image.Image:
    """Apply a circular low-pass frequency mask to a grayscale image."""
    if not math.isfinite(cutoff) or not 0.02 <= cutoff <= 0.5:
        raise ValueError("cutoff must be between 0.02 and 0.5")

    pixels = np.asarray(image.convert("L"), dtype=np.float64)
    height, width = pixels.shape
    spectrum = np.fft.fftshift(np.fft.fft2(pixels, axes=(0, 1)), axes=(0, 1))
    vertical = np.fft.fftshift(np.fft.fftfreq(height))
    horizontal = np.fft.fftshift(np.fft.fftfreq(width))
    radius_squared = vertical[:, None] ** 2 + horizontal[None, :] ** 2
    filtered_spectrum = spectrum * (radius_squared <= cutoff**2)
    reconstructed = np.fft.ifft2(
        np.fft.ifftshift(filtered_spectrum, axes=(0, 1)), axes=(0, 1)
    ).real
    return Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8))


def contact_sheet(images: dict[str, Image.Image]) -> Image.Image:
    """Build a two-column labeled sheet from the generated image variants."""
    if not images:
        raise ValueError("at least one image is required")
    tile_width, tile_height = 360, 290
    columns = 2
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (13, 20, 30))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (label, image) in enumerate(images.items()):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        draw.rounded_rectangle(
            (x + 8, y + 8, x + tile_width - 8, y + tile_height - 8),
            radius=8,
            fill=(22, 32, 45),
            outline=(52, 70, 91),
            width=1,
        )
        thumbnail = ImageOps.contain(
            image.convert("RGB"),
            (tile_width - 34, tile_height - 62),
            method=Image.Resampling.LANCZOS,
        )
        thumb_x = x + (tile_width - thumbnail.width) // 2
        thumb_y = y + 18 + (tile_height - 62 - thumbnail.height) // 2
        sheet.paste(thumbnail, (thumb_x, thumb_y))
        draw.text((x + 18, y + tile_height - 34), label, fill=(217, 229, 241), font=font)
    return sheet
