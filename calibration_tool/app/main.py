"""Tkinter entry point for the Homography Calibration Tool."""
from __future__ import annotations

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, simpledialog
from typing import Dict, Optional

import cv2

from .services.calibration import CalibrationManager, CalibrationSession
from .services.canvas import ensure_ground_plane, load_ground_plane_image, load_ground_plane_metadata
from .ui.events import (
    DisplayImage,
    canvas_click_to_image_point,
    draw_image,
    draw_placeholder,
    draw_points,
    draw_test_points,
    prepare_image_for_display,
)
from .ui.layout import (
    CAMERA_CANVAS_SIZE,
    GROUND_CANVAS_SIZE,
    create_root,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPORTS_DIR = ROOT_DIR / "exports"
DEFAULT_CAMERAS = ["Cam 1", "Cam 2", "Cam 3"]
STATUS_COLORS = {
    "info": "#1f6feb",
    "ok": "#2ea043",
    "warn": "#d29922",
    "error": "#f85149",
}


def format_point_list(points) -> str:
    if not points:
        return ""
    return "\n".join(f"{idx}: ({x:.1f}, {y:.1f})" for idx, (x, y) in enumerate(points, start=1))


def format_homography(matrix) -> str:
    if matrix is None:
        return ""
    rows = []
    for row in matrix:
        rows.append(" ".join(f"{value:9.4f}" for value in row))
    return "\n".join(rows)


def export_results(manager: CalibrationManager) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    export_path = EXPORTS_DIR / f"homography_{timestamp}.json"

    ground_meta = load_ground_plane_metadata().to_mapping()
    payload: Dict[str, object] = {
        "ground_plane": ground_meta,
        **manager.export_mapping(),
    }
    export_path.write_text(json.dumps(payload, indent=2))
    return export_path


def save_ground_overlays(manager: CalibrationManager) -> None:
    if not manager.has_any_homography():
        return

    base_image = load_ground_plane_image()
    export_folder = EXPORTS_DIR / "visualizations"
    export_folder.mkdir(parents=True, exist_ok=True)

    color_cycle = [
        (255, 128, 0),
        (0, 192, 255),
        (0, 255, 128),
        (255, 0, 128),
        (255, 255, 0),
    ]

    for index, session in enumerate(manager.sessions()):
        if not session.destination_points:
            continue
        image = base_image.copy()
        color = color_cycle[index % len(color_cycle)]
        for idx, (x, y) in enumerate(session.destination_points, start=1):
            cv2.circle(image, (int(round(x)), int(round(y))), 12, color, thickness=-1)
            cv2.putText(image, str(idx), (int(round(x)) + 10, int(round(y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        out_path = export_folder / f"ground_overlay_{session.camera_id.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), image)


class CalibrationApp:
    def __init__(self) -> None:
        ensure_ground_plane()
        self.manager = CalibrationManager()
        self.camera_options = DEFAULT_CAMERAS.copy()

        self.root, self.widgets = create_root(self.camera_options)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        ground_image = load_ground_plane_image()
        self.ground_display = prepare_image_for_display(ground_image, GROUND_CANVAS_SIZE)

        self.current_camera = self.widgets["camera_var"].get() or self.camera_options[0]
        self.current_camera_display: Optional[DisplayImage] = None
        self.test_click_pending = False

        self._setup_bindings()
        draw_placeholder(self.widgets["camera_canvas"], CAMERA_CANVAS_SIZE, "Load a camera frame")
        draw_image(self.widgets["ground_canvas"], self.ground_display)
        self._refresh_all(self.manager.get_session(self.current_camera))

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()

    # UI helpers -------------------------------------------------------
    def _set_status(self, message: str, level: str = "info") -> None:
        color = STATUS_COLORS.get(level, STATUS_COLORS["info"])
        status_var: tk.StringVar = self.widgets["status_var"]
        status_label: tk.Label = self.widgets["status_label"]
        status_var.set(message)
        status_label.configure(fg=color)

    def _update_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        if content:
            widget.insert(tk.END, content)
        widget.configure(state=tk.DISABLED)

    def _refresh_point_panels(self, session: CalibrationSession) -> None:
        self._update_text(self.widgets["source_text"], format_point_list(session.source_points))
        self._update_text(self.widgets["dest_text"], format_point_list(session.destination_points))
        self._update_text(self.widgets["homography_text"], format_homography(session.homography))

    def _refresh_buttons(self, camera_id: str) -> None:
        compute_state = tk.NORMAL if self.manager.can_compute_homography(camera_id) else tk.DISABLED
        test_state = tk.NORMAL if self.manager.get_session(camera_id).homography is not None else tk.DISABLED
        export_state = tk.NORMAL if self.manager.has_any_homography() else tk.DISABLED

        self.widgets["compute_button"].configure(state=compute_state)
        self.widgets["test_button"].configure(state=test_state)
        self.widgets["export_button"].configure(state=export_state)

    def _refresh_camera_view(self, session: CalibrationSession) -> None:
        canvas = self.widgets["camera_canvas"]
        if session.frame_image is None:
            draw_placeholder(canvas, CAMERA_CANVAS_SIZE, "Load a camera frame")
            self.current_camera_display = None
            return
        self.current_camera_display = prepare_image_for_display(session.frame_image, CAMERA_CANVAS_SIZE)
        draw_image(canvas, self.current_camera_display)
        draw_points(canvas, self.current_camera_display, session.source_points, color="#ffa657", label_prefix="S")

    def _refresh_ground_view(self, session: CalibrationSession) -> None:
        canvas = self.widgets["ground_canvas"]
        draw_image(canvas, self.ground_display)
        draw_points(canvas, self.ground_display, session.destination_points, color="#58a6ff", label_prefix="D")
        if session.test_points:
            draw_test_points(canvas, self.ground_display, session.test_points)

    def _refresh_all(self, session: CalibrationSession) -> None:
        self._refresh_camera_view(session)
        self._refresh_ground_view(session)
        self._refresh_point_panels(session)
        self._refresh_buttons(session.camera_id)

    def _setup_bindings(self) -> None:
        camera_var: tk.StringVar = self.widgets["camera_var"]
        camera_var.trace_add("write", self._on_camera_select)

        self.widgets["add_camera_button"].configure(command=self._on_add_camera)
        self.widgets["load_button"].configure(command=self._on_load_frame)
        self.widgets["undo_button"].configure(command=self._on_undo)
        self.widgets["reset_button"].configure(command=self._on_reset)
        self.widgets["compute_button"].configure(command=self._on_compute)
        self.widgets["test_button"].configure(command=self._on_test)
        self.widgets["export_button"].configure(command=self._on_export)

        self.widgets["camera_canvas"].bind("<Button-1>", self._on_camera_canvas_click)
        self.widgets["ground_canvas"].bind("<Button-1>", self._on_ground_canvas_click)

    # Event handlers ---------------------------------------------------
    def _on_add_camera(self) -> None:
        default_name = f"Cam {len(self.camera_options) + 1}"
        name = simpledialog.askstring("Add Camera", f"Camera name (default {default_name})")
        if name is None:
            return
        name = name.strip() or default_name
        if name in self.camera_options:
            self._set_status("Camera already exists", "warn")
            return
        self.camera_options.append(name)
        option_menu = self.widgets["camera_menu"]
        menu = option_menu["menu"]
        camera_var: tk.StringVar = self.widgets["camera_var"]
        menu.add_command(label=name, command=tk._setit(camera_var, name))
        camera_var.set(name)
        self.current_camera = name
        self.test_click_pending = False
        session = self.manager.get_session(name)
        self._refresh_all(session)
        self._set_status(f"Added camera {name}", "ok")

    def _on_camera_select(self, *_) -> None:
        selected = self.widgets["camera_var"].get()
        if not selected:
            return
        self.current_camera = selected
        self.test_click_pending = False
        self._refresh_all(self.manager.get_session(selected))
        self._set_status(f"Switched to {selected}")

    def _on_load_frame(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select camera frame",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if not file_path:
            return
        image = cv2.imread(file_path)
        if image is None:
            self._set_status("Failed to load image", "error")
            return
        self.manager.set_frame(self.current_camera, file_path, image)
        self.widgets["camera_file_var"].set(Path(file_path).name)
        self.test_click_pending = False
        session = self.manager.get_session(self.current_camera)
        self._refresh_all(session)
        self._set_status("Frame loaded", "ok")

    def _on_camera_canvas_click(self, event) -> None:
        session = self.manager.get_session(self.current_camera)
        if self.current_camera_display is None:
            self._set_status("Load a camera frame first", "warn")
            return
        point = canvas_click_to_image_point(event.x, event.y, self.current_camera_display)
        if point is None:
            self._set_status("Click inside the image area", "warn")
            return
        if self.test_click_pending:
            try:
                mapped_point = self.manager.transform_point(self.current_camera, point)
                self._refresh_ground_view(session)
                self._refresh_point_panels(session)
                self._set_status(f"Projected to ({mapped_point[0]:.1f}, {mapped_point[1]:.1f})", "ok")
            except ValueError as exc:
                self._set_status(str(exc), "error")
            finally:
                self.test_click_pending = False
                self._refresh_buttons(self.current_camera)
            return
        self.manager.add_source_point(self.current_camera, point)
        self._refresh_all(session)
        self._set_status(f"Captured source point #{len(session.source_points)}", "ok")

    def _on_ground_canvas_click(self, event) -> None:
        session = self.manager.get_session(self.current_camera)
        point = canvas_click_to_image_point(event.x, event.y, self.ground_display)
        if point is None:
            self._set_status("Click inside the map area", "warn")
            return
        try:
            self.manager.add_destination_point(self.current_camera, point)
            self._refresh_all(session)
            self._set_status(f"Captured map point #{len(session.destination_points)}", "ok")
        except ValueError as exc:
            self._set_status(str(exc), "error")

    def _on_undo(self) -> None:
        session = self.manager.get_session(self.current_camera)
        self.manager.undo_last_point(self.current_camera)
        self.test_click_pending = False
        self._refresh_all(session)
        self._set_status("Removed last point pair")

    def _on_reset(self) -> None:
        session = self.manager.get_session(self.current_camera)
        self.manager.reset_camera(self.current_camera)
        self.test_click_pending = False
        self._refresh_all(session)
        self._set_status("Cleared points", "warn")

    def _on_compute(self) -> None:
        try:
            self.manager.compute_homography(self.current_camera)
            self._refresh_point_panels(self.manager.get_session(self.current_camera))
            self._refresh_buttons(self.current_camera)
            self._set_status("Homography computed", "ok")
        except (ValueError, RuntimeError) as exc:
            self._set_status(str(exc), "error")

    def _on_test(self) -> None:
        if self.manager.get_session(self.current_camera).homography is None:
            self._set_status("Compute homography first", "warn")
            return
        self.test_click_pending = True
        self._set_status("Click on the camera image to test")

    def _on_export(self) -> None:
        if not self.manager.has_any_homography():
            self._set_status("No homographies to export", "warn")
            return
        export_path = export_results(self.manager)
        save_ground_overlays(self.manager)
        self._set_status(f"Exported to {export_path.name}", "ok")


def run() -> None:
    app = CalibrationApp()
    app.run()


if __name__ == "__main__":
    run()
