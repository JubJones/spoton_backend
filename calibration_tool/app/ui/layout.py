"""Tkinter layout helpers for the calibration tool."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Sequence, Tuple

CAMERA_CANVAS_SIZE: Tuple[int, int] = (640, 480)
GROUND_CANVAS_SIZE: Tuple[int, int] = (640, 480)


def create_root(camera_options: Sequence[str]) -> tuple[tk.Tk, dict[str, object]]:
    root = tk.Tk()
    root.title("Homography Calibration Tool")
    root.resizable(False, False)

    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    main_frame = ttk.Frame(root, padding=10)
    main_frame.grid(row=0, column=0)

    # Camera column --------------------------------------------------
    camera_frame = ttk.Frame(main_frame)
    camera_frame.grid(row=0, column=0, sticky="n")

    camera_var = tk.StringVar(value=camera_options[0] if camera_options else "")
    camera_menu = ttk.OptionMenu(camera_frame, camera_var, camera_var.get(), *camera_options)
    camera_menu.grid(row=0, column=0, padx=(0, 6), pady=(0, 6))

    add_camera_button = ttk.Button(camera_frame, text="Add Camera")
    add_camera_button.grid(row=0, column=1, pady=(0, 6))

    load_button = ttk.Button(camera_frame, text="Load Camera Frame")
    load_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

    camera_file_var = tk.StringVar(value="No frame loaded")
    camera_file_label = ttk.Label(camera_frame, textvariable=camera_file_var, width=32)
    camera_file_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

    camera_canvas = tk.Canvas(camera_frame, width=CAMERA_CANVAS_SIZE[0], height=CAMERA_CANVAS_SIZE[1], bg="black")
    camera_canvas.grid(row=3, column=0, columnspan=2)

    # Ground column --------------------------------------------------
    ground_frame = ttk.Frame(main_frame, padding=(20, 0))
    ground_frame.grid(row=0, column=1, sticky="n")

    ground_label = ttk.Label(ground_frame, text="Ground Plane")
    ground_label.grid(row=0, column=0, pady=(0, 6))

    ground_canvas = tk.Canvas(ground_frame, width=GROUND_CANVAS_SIZE[0], height=GROUND_CANVAS_SIZE[1], bg="black")
    ground_canvas.grid(row=1, column=0)

    # Side panel -----------------------------------------------------
    side_frame = ttk.Frame(main_frame, padding=(20, 0))
    side_frame.grid(row=0, column=2, sticky="n")

    ttk.Label(side_frame, text="Source Points").grid(row=0, column=0, sticky="w")
    ttk.Label(side_frame, text="Destination Points").grid(row=0, column=1, sticky="w", padx=(10, 0))

    source_list = tk.Listbox(side_frame, width=28, height=12, exportselection=False)
    dest_list = tk.Listbox(side_frame, width=28, height=12, exportselection=False)
    source_list.grid(row=1, column=0, pady=(0, 4))
    dest_list.grid(row=1, column=1, pady=(0, 4), padx=(10, 0))

    edit_source_button = ttk.Button(side_frame, text="Edit Source")
    edit_dest_button = ttk.Button(side_frame, text="Edit Map")
    edit_source_button.grid(row=2, column=0, sticky="ew", pady=(0, 2))
    edit_dest_button.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(0, 2))

    delete_source_button = ttk.Button(side_frame, text="Delete Source")
    delete_dest_button = ttk.Button(side_frame, text="Delete Map")
    delete_source_button.grid(row=3, column=0, sticky="ew", pady=(0, 6))
    delete_dest_button.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=(0, 6))

    ttk.Label(side_frame, text="Homography Matrix").grid(row=4, column=0, columnspan=2, sticky="w")
    homography_text = tk.Text(side_frame, width=38, height=8, state="disabled")
    homography_text.grid(row=5, column=0, columnspan=2, pady=(0, 8))

    undo_button = ttk.Button(side_frame, text="Undo Last Source")
    reset_button = ttk.Button(side_frame, text="Reset Points")
    undo_button.grid(row=6, column=0, sticky="ew", pady=(0, 4))
    reset_button.grid(row=6, column=1, sticky="ew", padx=(10, 0), pady=(0, 4))

    compute_button = ttk.Button(side_frame, text="Compute Homography", state=tk.DISABLED)
    compute_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 4))

    test_button = ttk.Button(side_frame, text="Test Transform", state=tk.DISABLED)
    test_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 4))

    export_button = ttk.Button(side_frame, text="Export", state=tk.DISABLED)
    export_button.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 4))

    zones_button = ttk.Button(side_frame, text="Zones (coming soon)", state=tk.DISABLED)
    zones_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(0, 8))

    status_var = tk.StringVar(value="Ready")
    status_label = tk.Label(side_frame, textvariable=status_var, fg="#1f6feb")
    status_label.grid(row=11, column=0, columnspan=2, sticky="w")

    widgets: dict[str, object] = {
        "camera_var": camera_var,
        "camera_menu": camera_menu,
        "add_camera_button": add_camera_button,
        "load_button": load_button,
        "camera_file_var": camera_file_var,
        "camera_canvas": camera_canvas,
        "ground_canvas": ground_canvas,
        "source_list": source_list,
        "dest_list": dest_list,
        "edit_source_button": edit_source_button,
        "edit_dest_button": edit_dest_button,
        "delete_source_button": delete_source_button,
        "delete_dest_button": delete_dest_button,
        "homography_text": homography_text,
        "undo_button": undo_button,
        "reset_button": reset_button,
        "compute_button": compute_button,
        "test_button": test_button,
        "export_button": export_button,
        "zones_button": zones_button,
        "status_var": status_var,
        "status_label": status_label,
    }

    return root, widgets
