"""Utility helpers for Tkinter canvas rendering and coordinate transforms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
from PIL import Image, ImageTk

Point = Tuple[float, float]


@dataclass
class DisplayImage:
    photo: ImageTk.PhotoImage
    canvas_size: Tuple[int, int]
    displayed_size: Tuple[int, int]
    scale: float
    offset: Tuple[int, int]
    original_size: Tuple[int, int]


def prepare_image_for_display(image: np.ndarray, canvas_size: Tuple[int, int]) -> DisplayImage:
    if image is None:
        raise ValueError("No image provided for display preparation")

    canvas_width, canvas_height = canvas_size
    original_height, original_width = image.shape[:2]
    scale = min(canvas_width / original_width, canvas_height / original_height)
    display_width = max(1, int(original_width * scale))
    display_height = max(1, int(original_height * scale))
    resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    offset_x = (canvas_width - display_width) // 2
    offset_y = (canvas_height - display_height) // 2

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    photo = ImageTk.PhotoImage(image=pil_image)

    return DisplayImage(
        photo=photo,
        canvas_size=canvas_size,
        displayed_size=(display_width, display_height),
        scale=scale,
        offset=(offset_x, offset_y),
        original_size=(original_width, original_height),
    )


def draw_placeholder(canvas, canvas_size: Tuple[int, int], message: str) -> None:
    canvas.delete("all")
    canvas.create_rectangle(0, 0, canvas_size[0], canvas_size[1], fill="#1f1f1f", outline="")
    canvas.create_text(
        canvas_size[0] // 2,
        canvas_size[1] // 2,
        text=message,
        fill="white",
        font=("Helvetica", 14),
    )


def draw_image(canvas, display_image: DisplayImage) -> None:
    canvas.delete("all")
    canvas.create_rectangle(0, 0, display_image.canvas_size[0], display_image.canvas_size[1], fill="black", outline="")
    canvas.create_image(display_image.offset[0], display_image.offset[1], anchor="nw", image=display_image.photo)
    canvas.image = display_image.photo  # prevent garbage collection


def _display_coordinates(point: Point, display_image: DisplayImage) -> Tuple[float, float]:
    display_x = display_image.offset[0] + point[0] * display_image.scale
    display_y = display_image.offset[1] + point[1] * display_image.scale
    return display_x, display_y


def draw_points(
    canvas,
    display_image: DisplayImage,
    points: Sequence[Point],
    color: str,
    radius: int = 6,
    label_prefix: str = "",
) -> None:
    for idx, point in enumerate(points, start=1):
        x, y = _display_coordinates(point, display_image)
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        label = f"{label_prefix}{idx}" if label_prefix else str(idx)
        canvas.create_text(x + radius + 4, y - radius - 4, text=label, fill=color, font=("Helvetica", 10, "bold"))


def draw_test_points(
    canvas,
    display_image: DisplayImage,
    points: Iterable[Point],
    color: str = "cyan",
    radius: int = 5,
) -> None:
    for point in points:
        x, y = _display_coordinates(point, display_image)
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color)


def canvas_click_to_image_point(event_x: float, event_y: float, display_image: DisplayImage) -> Optional[Point]:
    if display_image is None:
        return None

    offset_x, offset_y = display_image.offset
    width, height = display_image.displayed_size

    if not (offset_x <= event_x <= offset_x + width):
        return None
    if not (offset_y <= event_y <= offset_y + height):
        return None

    normalized_x = (event_x - offset_x) / display_image.scale
    normalized_y = (event_y - offset_y) / display_image.scale
    return float(normalized_x), float(normalized_y)
