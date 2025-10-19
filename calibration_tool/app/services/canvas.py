"""Utility helpers for managing the calibration ground plane assets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Paths --------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = _BASE_DIR / "assets"
GROUND_PLANE_IMAGE = ASSETS_DIR / "ground_plane_map.png"
GROUND_PLANE_META = ASSETS_DIR / "ground_plane.json"


@dataclass(frozen=True)
class GroundPlaneMetadata:
    """Lightweight container describing the ground plane image."""

    width: int
    height: int
    pixels_per_meter: int
    image_path: str

    @classmethod
    def from_mapping(cls, data: Dict[str, int]) -> "GroundPlaneMetadata":
        return cls(
            width=int(data.get("width", 2000)),
            height=int(data.get("height", 1500)),
            pixels_per_meter=int(data.get("pixels_per_meter", 100)),
            image_path=str(data.get("image_path", GROUND_PLANE_IMAGE.as_posix())),
        )

    def to_mapping(self) -> Dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "pixels_per_meter": self.pixels_per_meter,
            "image_path": self.image_path,
        }


def create_blank_canvas(
    width: int = 2000,
    height: int = 1500,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a blank RGB canvas image."""

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = color
    return canvas


def ensure_ground_plane(
    width: int = 2000,
    height: int = 1500,
    color: Tuple[int, int, int] = (0, 0, 0),
    pixels_per_meter: int = 100,
) -> GroundPlaneMetadata:
    """Ensure the default ground plane image and metadata exist on disk."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    metadata: GroundPlaneMetadata
    if GROUND_PLANE_META.exists():
        metadata = GroundPlaneMetadata.from_mapping(json.loads(GROUND_PLANE_META.read_text()))
    else:
        metadata = GroundPlaneMetadata(width=width, height=height, pixels_per_meter=pixels_per_meter, image_path=str(GROUND_PLANE_IMAGE))
        GROUND_PLANE_META.write_text(json.dumps(metadata.to_mapping(), indent=2))

    if not GROUND_PLANE_IMAGE.exists():
        blank = create_blank_canvas(metadata.width, metadata.height, color)
        cv2.imwrite(str(GROUND_PLANE_IMAGE), blank)

    return metadata


def load_ground_plane_image() -> np.ndarray:
    """Load the ground plane image as a NumPy array."""

    ensure_ground_plane()
    image = cv2.imread(str(GROUND_PLANE_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Unable to load ground plane image at {GROUND_PLANE_IMAGE}")
    return image


def load_ground_plane_metadata() -> GroundPlaneMetadata:
    """Load metadata for the ground plane, creating defaults if missing."""

    meta = ensure_ground_plane()
    return meta
