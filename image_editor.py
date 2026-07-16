"""
Image Editor for YOLO & CNN Number Plate Recognition
=====================================================
A GUI tool for:
  - Annotating images with YOLO-format bounding boxes
  - Preprocessing / augmenting images for CNN training
  - Batch exporting processed images and annotations

Requirements: Pillow  (pip install pillow)
"""

import math
import os
import random
import json
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
DEFAULT_CLASS = "number_plate"
CANVAS_MAX_W = 700
CANVAS_MAX_H = 600
ANGLE_EPSILON = 0.01   # degrees; below this an OBB angle is treated as zero

YOLO_SIZES = [
    ("416 × 416 (YOLOv3/v4)", 416, 416),
    ("640 × 640 (YOLOv5/v8)", 640, 640),
    ("1280 × 1280 (YOLOv8 large)", 1280, 1280),
]
CNN_SIZES = [
    ("224 × 224 (VGG / ResNet)", 224, 224),
    ("128 × 128 (LightCNN)", 128, 128),
    ("64 × 64 (small CNN)", 64, 64),
]


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def canvas_to_image_coords(cx, cy, offset_x, offset_y, scale):
    """Convert canvas pixel (cx, cy) → original image pixel."""
    return (cx - offset_x) / scale, (cy - offset_y) / scale


def image_to_canvas_coords(ix, iy, offset_x, offset_y, scale):
    """Convert original image pixel → canvas pixel."""
    return ix * scale + offset_x, iy * scale + offset_y


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel bounding box → YOLO normalised format."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = abs(x2 - x1) / img_w
    bh = abs(y2 - y1) / img_h
    return cx, cy, bw, bh


def yolo_to_bbox(cx, cy, bw, bh, img_w, img_h):
    """Convert YOLO normalised format → pixel bounding box."""
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────────────────────────
# OBB helpers
# ──────────────────────────────────────────────────────────────────────────────
def rotate_point(px, py, cx, cy, angle_deg):
    """Rotate pixel (px, py) around centre (cx, cy) by angle_deg degrees."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return dx * cos_a - dy * sin_a + cx, dx * sin_a + dy * cos_a + cy


def bbox_to_obb_corners(x1, y1, x2, y2, angle_deg=0.0):
    """Return the 4 corners of an oriented bounding box (top-left → clockwise)."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if angle_deg == 0.0:
        return corners
    return [rotate_point(px, py, cx, cy, angle_deg) for px, py in corners]


def obb_corners_to_yolo(corners, img_w, img_h):
    """Normalise 4 corner points for YOLO OBB format (8 floats)."""
    flat = []
    for px, py in corners:
        flat.extend([px / img_w, py / img_h])
    return flat


def corners_to_bbox_angle(corners):
    """Reconstruct axis-aligned bbox and rotation angle from 4 OBB corners."""
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle_deg = math.degrees(math.atan2(dy, dx))
    unrotated = [rotate_point(px, py, cx, cy, -angle_deg) for px, py in corners]
    x1 = min(p[0] for p in unrotated)
    y1 = min(p[1] for p in unrotated)
    x2 = max(p[0] for p in unrotated)
    y2 = max(p[1] for p in unrotated)
    return x1, y1, x2, y2, angle_deg


# ──────────────────────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────────────────────
class ImageEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Number Plate Image Editor – YOLO & CNN")
        self.geometry("1280x780")
        self.minsize(900, 600)
        self.configure(bg="#2b2b2b")

        # ── State ──────────────────────────────────────────────────────────
        self.image_paths: list[str] = []
        self.current_index: int = -1
        self.original_image: Image.Image | None = None   # untouched original
        self.display_image: Image.Image | None = None    # after transforms
        self.tk_image: ImageTk.PhotoImage | None = None

        # canvas rendering helpers
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.canvas_scale = 1.0

        # annotations: dict[path] → list of {"class": str, "bbox": [x1,y1,x2,y2]}
        self.annotations: dict[str, list] = {}
        # dimensions of the display_image when annotations were drawn (may differ
        # from original due to rotation with expand=True)
        self.annotation_img_sizes: dict[str, tuple[int, int]] = {}
        self.class_list: list[str] = [DEFAULT_CLASS]

        # drawing state
        self.drawing = False
        self.draw_start_x = 0
        self.draw_start_y = 0
        self.temp_rect_id = None

        # transform controls (per-image; reset on navigate)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sharpness_var = tk.DoubleVar(value=1.0)
        self.grayscale_var = tk.BooleanVar(value=False)
        self.flip_h_var = tk.BooleanVar(value=False)
        self.flip_v_var = tk.BooleanVar(value=False)
        self.rotate_var = tk.IntVar(value=0)  # 0, 90, 180, 270

        # output settings
        self.output_dir: str = ""
        self.export_mode = tk.StringVar(value="yolo")   # "yolo" | "yolo_obb" | "cnn"
        self.resize_mode = tk.StringVar(value="none")   # "none" | preset key
        self.custom_w = tk.IntVar(value=640)
        self.custom_h = tk.IntVar(value=640)

        # OBB (Oriented Bounding Box) state
        self.obb_mode_var = tk.BooleanVar(value=False)
        self.obb_angle_var = tk.DoubleVar(value=0.0)

        # Train/Val/Test split state
        self.split_var = tk.BooleanVar(value=True)
        self.split_train = tk.IntVar(value=70)
        self.split_val = tk.IntVar(value=20)
        self.split_test = tk.IntVar(value=10)

        self._build_ui()
        self._bind_keys()

    # ──────────────────────────────────────────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0")
        style.configure("TButton", padding=4)
        style.configure("TCheckbutton", background="#2b2b2b", foreground="#e0e0e0")
        style.configure("TLabelframe", background="#2b2b2b", foreground="#aaaaaa")
        style.configure("TLabelframe.Label", background="#2b2b2b", foreground="#aaaaaa")
        style.configure("TScale", background="#2b2b2b")
        style.configure("TCombobox", fieldbackground="#3c3c3c", foreground="#e0e0e0")

        # ── Top toolbar ────────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Button(toolbar, text="📂 Open Images", command=self.open_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📁 Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="◀ Prev", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next ▶", command=self.next_image).pack(side=tk.LEFT, padx=2)
        self.img_counter_lbl = ttk.Label(toolbar, text="No images loaded")
        self.img_counter_lbl.pack(side=tk.LEFT, padx=8)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="💾 Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📤 Batch Export", command=self.batch_export).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🗑 Clear Boxes", command=self.clear_boxes).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="🤖 Auto-Annotate", command=self.auto_annotate).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🤖 Auto-Annotate All", command=self.auto_annotate_all).pack(side=tk.LEFT, padx=2)

        # ── Main area ──────────────────────────────────────────────────────
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: image list
        left_panel = ttk.LabelFrame(main, text="Images", width=160)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left_panel.pack_propagate(False)
        self.img_listbox = tk.Listbox(
            left_panel, bg="#3c3c3c", fg="#e0e0e0",
            selectbackground="#555577", activestyle="none",
            font=("Consolas", 9), width=20,
        )
        self.img_listbox.pack(fill=tk.BOTH, expand=True)
        self.img_listbox.bind("<<ListboxSelect>>", self._on_list_select)
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.img_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_listbox.configure(yscrollcommand=scrollbar.set)

        # Center: canvas
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            center, bg="#1e1e1e", cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Configure>", lambda e: self._refresh_canvas())

        # ── Status bar ─────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready. Open images to start.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Right: controls
        right_panel = ttk.Frame(main, width=270)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_panel.pack_propagate(False)
        self._build_right_panel(right_panel)

    def _build_right_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # ── Tab 1: Annotation ──────────────────────────────────────────────
        ann_tab = ttk.Frame(notebook)
        notebook.add(ann_tab, text="Annotate")

        ttk.Label(ann_tab, text="Active Class:").pack(anchor=tk.W, padx=6, pady=(8, 0))
        self.class_combo = ttk.Combobox(ann_tab, values=self.class_list, state="readonly", width=22)
        self.class_combo.set(DEFAULT_CLASS)
        self.class_combo.pack(anchor=tk.W, padx=6, pady=2)

        class_btn_row = ttk.Frame(ann_tab)
        class_btn_row.pack(anchor=tk.W, padx=6, pady=2)
        ttk.Button(class_btn_row, text="+ Add Class", command=self._add_class).pack(side=tk.LEFT)
        ttk.Button(class_btn_row, text="- Remove", command=self._remove_class).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ann_tab).pack(fill=tk.X, pady=6)
        ttk.Label(ann_tab, text="Bounding Boxes:").pack(anchor=tk.W, padx=6)

        box_frame = ttk.Frame(ann_tab)
        box_frame.pack(fill=tk.BOTH, expand=True, padx=6)
        self.box_listbox = tk.Listbox(
            box_frame, bg="#3c3c3c", fg="#e0e0e0",
            selectbackground="#555577", activestyle="none",
            font=("Consolas", 8), height=10,
        )
        self.box_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Button(ann_tab, text="🗑 Delete Selected Box", command=self._delete_selected_box).pack(
            anchor=tk.W, padx=6, pady=4
        )
        ttk.Button(ann_tab, text="📋 Load Annotations", command=self._load_annotations_for_current).pack(
            anchor=tk.W, padx=6, pady=2
        )

        ttk.Separator(ann_tab).pack(fill=tk.X, pady=6)
        ttk.Label(ann_tab, text="OBB (Oriented Box):", foreground="#aaaaaa").pack(anchor=tk.W, padx=6)
        ttk.Checkbutton(ann_tab, text="OBB Mode", variable=self.obb_mode_var).pack(
            anchor=tk.W, padx=6, pady=2
        )
        angle_row = ttk.Frame(ann_tab)
        angle_row.pack(anchor=tk.W, padx=6, pady=2)
        ttk.Label(angle_row, text="Angle (°):").pack(side=tk.LEFT)
        ttk.Spinbox(
            angle_row, from_=-180, to=180, increment=1,
            textvariable=self.obb_angle_var, width=7,
        ).pack(side=tk.LEFT, padx=4)

        # ── Tab 2: Transforms ─────────────────────────────────────────────
        tf_tab = ttk.Frame(notebook)
        notebook.add(tf_tab, text="Transforms")

        def _slider(parent, label, var, from_, to, res, cmd):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            s = ttk.Scale(row, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=cmd)
            s.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = ttk.Label(row, width=5)
            lbl.pack(side=tk.LEFT)

            def _update_lbl(*_):
                lbl.config(text=f"{var.get():.2f}")

            var.trace_add("write", _update_lbl)
            _update_lbl()

        ttk.Label(tf_tab, text="Colour Adjustments", foreground="#aaaaaa").pack(anchor=tk.W, padx=6, pady=(8, 0))
        _slider(tf_tab, "Brightness", self.brightness_var, 0.1, 3.0, 0.05, lambda _: self._apply_transforms())
        _slider(tf_tab, "Contrast", self.contrast_var, 0.1, 3.0, 0.05, lambda _: self._apply_transforms())
        _slider(tf_tab, "Sharpness", self.sharpness_var, 0.0, 5.0, 0.1, lambda _: self._apply_transforms())

        ttk.Separator(tf_tab).pack(fill=tk.X, pady=6)
        ttk.Label(tf_tab, text="Geometric", foreground="#aaaaaa").pack(anchor=tk.W, padx=6)

        flip_row = ttk.Frame(tf_tab)
        flip_row.pack(anchor=tk.W, padx=6, pady=2)
        ttk.Checkbutton(flip_row, text="Flip Horizontal", variable=self.flip_h_var,
                        command=self._apply_transforms).pack(side=tk.LEFT)
        ttk.Checkbutton(flip_row, text="Flip Vertical", variable=self.flip_v_var,
                        command=self._apply_transforms).pack(side=tk.LEFT, padx=8)

        rot_row = ttk.Frame(tf_tab)
        rot_row.pack(anchor=tk.W, padx=6, pady=2)
        ttk.Label(rot_row, text="Rotate:").pack(side=tk.LEFT)
        for deg in [0, 90, 180, 270]:
            ttk.Radiobutton(rot_row, text=str(deg), variable=self.rotate_var,
                            value=deg, command=self._apply_transforms).pack(side=tk.LEFT)

        ttk.Separator(tf_tab).pack(fill=tk.X, pady=6)
        ttk.Checkbutton(tf_tab, text="Grayscale", variable=self.grayscale_var,
                        command=self._apply_transforms).pack(anchor=tk.W, padx=6)

        ttk.Button(tf_tab, text="↺ Reset Transforms", command=self._reset_transforms).pack(
            anchor=tk.W, padx=6, pady=8
        )

        # ── Tab 3: Export ─────────────────────────────────────────────────
        exp_tab = ttk.Frame(notebook)
        notebook.add(exp_tab, text="Export")

        ttk.Label(exp_tab, text="Export Mode:", foreground="#aaaaaa").pack(anchor=tk.W, padx=6, pady=(8, 0))
        ttk.Radiobutton(exp_tab, text="YOLO  (images + .txt labels)",
                        variable=self.export_mode, value="yolo").pack(anchor=tk.W, padx=12)
        ttk.Radiobutton(exp_tab, text="YOLO OBB  (images + .txt OBB labels)",
                        variable=self.export_mode, value="yolo_obb").pack(anchor=tk.W, padx=12)
        ttk.Radiobutton(exp_tab, text="CNN   (images only, organised)",
                        variable=self.export_mode, value="cnn").pack(anchor=tk.W, padx=12)

        ttk.Separator(exp_tab).pack(fill=tk.X, pady=6)
        ttk.Label(exp_tab, text="Resize:", foreground="#aaaaaa").pack(anchor=tk.W, padx=6)

        self.resize_mode.set("none")
        ttk.Radiobutton(exp_tab, text="No resize", variable=self.resize_mode, value="none").pack(anchor=tk.W, padx=12)
        for label, w, h in YOLO_SIZES:
            ttk.Radiobutton(exp_tab, text=label, variable=self.resize_mode,
                            value=f"{w}x{h}").pack(anchor=tk.W, padx=12)
        for label, w, h in CNN_SIZES:
            ttk.Radiobutton(exp_tab, text=label, variable=self.resize_mode,
                            value=f"{w}x{h}").pack(anchor=tk.W, padx=12)

        custom_row = ttk.Frame(exp_tab)
        custom_row.pack(anchor=tk.W, padx=6, pady=4)
        ttk.Radiobutton(custom_row, text="Custom:", variable=self.resize_mode,
                        value="custom").pack(side=tk.LEFT)
        ttk.Entry(custom_row, textvariable=self.custom_w, width=5).pack(side=tk.LEFT)
        ttk.Label(custom_row, text="×").pack(side=tk.LEFT)
        ttk.Entry(custom_row, textvariable=self.custom_h, width=5).pack(side=tk.LEFT)

        ttk.Separator(exp_tab).pack(fill=tk.X, pady=6)
        ttk.Label(exp_tab, text="Train/Val/Test Split:", foreground="#aaaaaa").pack(anchor=tk.W, padx=6)
        ttk.Checkbutton(exp_tab, text="Split into train/val/test",
                        variable=self.split_var).pack(anchor=tk.W, padx=12, pady=(2, 0))
        split_row = ttk.Frame(exp_tab)
        split_row.pack(anchor=tk.W, padx=12, pady=2)
        ttk.Label(split_row, text="Train %:").pack(side=tk.LEFT)
        ttk.Spinbox(split_row, from_=1, to=98, textvariable=self.split_train, width=4).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(split_row, text="Val %:").pack(side=tk.LEFT)
        ttk.Spinbox(split_row, from_=1, to=98, textvariable=self.split_val, width=4).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(split_row, text="Test %:").pack(side=tk.LEFT)
        ttk.Spinbox(split_row, from_=0, to=98, textvariable=self.split_test, width=4).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Separator(exp_tab).pack(fill=tk.X, pady=6)
        dir_row = ttk.Frame(exp_tab)
        dir_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(dir_row, text="Output dir:").pack(side=tk.LEFT)
        ttk.Button(dir_row, text="Choose…", command=self._choose_output_dir).pack(side=tk.RIGHT)
        self.output_dir_lbl = ttk.Label(exp_tab, text="(not set)", foreground="#888888",
                                        wraplength=240, justify=tk.LEFT)
        self.output_dir_lbl.pack(anchor=tk.W, padx=6, pady=2)

        ttk.Button(exp_tab, text="📤 Batch Export All", command=self.batch_export).pack(
            anchor=tk.W, padx=6, pady=8
        )

    def _bind_keys(self):
        self.bind("<Left>", lambda _: self.prev_image())
        self.bind("<Right>", lambda _: self.next_image())
        self.bind("<Delete>", lambda _: self._delete_selected_box())

    # ──────────────────────────────────────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────────────────────────────────────
    def open_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", " ".join(f"*{e}" for e in SUPPORTED_EXTS)),
                       ("All files", "*.*")],
        )
        if paths:
            self._load_paths(list(paths))

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            paths = sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(SUPPORTED_EXTS)
            )
            if paths:
                self._load_paths(paths)
            else:
                messagebox.showinfo("No images", "No supported images found in that folder.")

    def _load_paths(self, paths: list[str]):
        self.image_paths = paths
        self.img_listbox.delete(0, tk.END)
        for p in paths:
            self.img_listbox.insert(tk.END, os.path.basename(p))
        self.current_index = 0
        self._show_current()

    def _show_current(self):
        if not self.image_paths or self.current_index < 0:
            return
        path = self.image_paths[self.current_index]
        try:
            self.original_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open image:\n{exc}")
            return

        self._reset_transforms(refresh=False)
        self._apply_transforms()
        self._update_counter()
        self._update_box_listbox()

        self.img_listbox.selection_clear(0, tk.END)
        self.img_listbox.selection_set(self.current_index)
        self.img_listbox.see(self.current_index)

        self.status_var.set(f"Loaded: {path}")

    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._show_current()

    def _on_list_select(self, _event):
        sel = self.img_listbox.curselection()
        if sel:
            self.current_index = sel[0]
            self._show_current()

    def _update_counter(self):
        if self.image_paths:
            self.img_counter_lbl.config(
                text=f"{self.current_index + 1} / {len(self.image_paths)}"
            )
        else:
            self.img_counter_lbl.config(text="No images loaded")

    # ──────────────────────────────────────────────────────────────────────────
    # Transforms
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_transforms(self, refresh=True):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.sharpness_var.set(1.0)
        self.grayscale_var.set(False)
        self.flip_h_var.set(False)
        self.flip_v_var.set(False)
        self.rotate_var.set(0)
        if refresh:
            self._apply_transforms()

    def _apply_transforms(self, *_):
        if self.original_image is None:
            return
        img = self.original_image.copy()

        # colour
        img = ImageEnhance.Brightness(img).enhance(self.brightness_var.get())
        img = ImageEnhance.Contrast(img).enhance(self.contrast_var.get())
        img = ImageEnhance.Sharpness(img).enhance(self.sharpness_var.get())

        # grayscale
        if self.grayscale_var.get():
            img = ImageOps.grayscale(img).convert("RGB")

        # flips
        if self.flip_h_var.get():
            img = ImageOps.mirror(img)
        if self.flip_v_var.get():
            img = ImageOps.flip(img)

        # rotation
        deg = self.rotate_var.get()
        if deg:
            img = img.rotate(-deg, expand=True)

        self.display_image = img
        self._refresh_canvas()

    # ──────────────────────────────────────────────────────────────────────────
    # Canvas rendering
    # ──────────────────────────────────────────────────────────────────────────
    def _refresh_canvas(self):
        self.canvas.delete("all")
        if self.display_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        iw, ih = self.display_image.size
        scale = min(cw / iw, ch / ih, 1.0)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        self.canvas_scale = scale
        self.canvas_offset_x = (cw - new_w) // 2
        self.canvas_offset_y = (ch - new_h) // 2

        resized = self.display_image.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                 anchor=tk.NW, image=self.tk_image)

        # draw saved boxes
        path = self._current_path()
        if path and path in self.annotations:
            iw_orig, ih_orig = self.original_image.size
            for i, ann in enumerate(self.annotations[path]):
                x1, y1, x2, y2 = ann["bbox"]
                angle = ann.get("angle", 0.0)
                color = "#00ff88"
                tag = f"box_{i}"
                if angle != 0.0:
                    corners = bbox_to_obb_corners(x1, y1, x2, y2, angle)
                    canvas_corners = [
                        image_to_canvas_coords(px, py, self.canvas_offset_x, self.canvas_offset_y, scale)
                        for px, py in corners
                    ]
                    flat = [coord for pt in canvas_corners for coord in pt]
                    self.canvas.create_polygon(flat, outline=color, fill="", width=2, tags=tag)
                    lx, ly = canvas_corners[0]
                    self.canvas.create_text(
                        lx + 2, ly - 10,
                        text=f"{ann['class']} ({angle:.1f}°)",
                        fill=color, anchor=tk.W, font=("Helvetica", 9, "bold"),
                    )
                else:
                    cx1, cy1 = image_to_canvas_coords(x1, y1, self.canvas_offset_x, self.canvas_offset_y, scale)
                    cx2, cy2 = image_to_canvas_coords(x2, y2, self.canvas_offset_x, self.canvas_offset_y, scale)
                    self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, tags=tag)
                    self.canvas.create_text(cx1 + 2, cy1 - 10, text=ann["class"],
                                            fill=color, anchor=tk.W, font=("Helvetica", 9, "bold"))

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing bounding boxes
    # ──────────────────────────────────────────────────────────────────────────
    def _on_mouse_press(self, event):
        if self.display_image is None:
            return
        self.drawing = True
        self.draw_start_x = event.x
        self.draw_start_y = event.y
        self.temp_rect_id = None

    def _on_mouse_drag(self, event):
        if not self.drawing:
            return
        if self.temp_rect_id:
            self.canvas.delete(self.temp_rect_id)
        self.temp_rect_id = self.canvas.create_rectangle(
            self.draw_start_x, self.draw_start_y, event.x, event.y,
            outline="#ffaa00", width=2, dash=(4, 2),
        )

    def _on_mouse_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        if self.temp_rect_id:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        cx1 = min(self.draw_start_x, event.x)
        cy1 = min(self.draw_start_y, event.y)
        cx2 = max(self.draw_start_x, event.x)
        cy2 = max(self.draw_start_y, event.y)

        # ignore tiny accidental clicks
        if abs(cx2 - cx1) < 5 or abs(cy2 - cy1) < 5:
            return

        ix1, iy1 = canvas_to_image_coords(cx1, cy1, self.canvas_offset_x, self.canvas_offset_y, self.canvas_scale)
        ix2, iy2 = canvas_to_image_coords(cx2, cy2, self.canvas_offset_x, self.canvas_offset_y, self.canvas_scale)

        iw, ih = self.display_image.size
        ix1 = clamp(ix1, 0, iw)
        iy1 = clamp(iy1, 0, ih)
        ix2 = clamp(ix2, 0, iw)
        iy2 = clamp(iy2, 0, ih)

        path = self._current_path()
        if path is None:
            return
        if path not in self.annotations:
            self.annotations[path] = []

        cls = self.class_combo.get() or DEFAULT_CLASS
        ann = {"class": cls, "bbox": [ix1, iy1, ix2, iy2]}
        if self.obb_mode_var.get():
            try:
                angle = float(self.obb_angle_var.get())
            except (ValueError, tk.TclError):
                angle = 0.0
            if angle != 0.0:
                ann["angle"] = angle
        self.annotations[path].append(ann)
        # track display_image dimensions so YOLO conversion uses the correct size
        self.annotation_img_sizes[path] = self.display_image.size
        self._update_box_listbox()
        self._refresh_canvas()
        angle_info = f"  angle={ann.get('angle', 0.0):.1f}°" if "angle" in ann else ""
        self.status_var.set(
            f"Box added → class '{cls}'  [{ix1:.0f},{iy1:.0f},{ix2:.0f},{iy2:.0f}]{angle_info}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Annotation management
    # ──────────────────────────────────────────────────────────────────────────
    def _current_path(self) -> str | None:
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def _update_box_listbox(self):
        self.box_listbox.delete(0, tk.END)
        path = self._current_path()
        if path and path in self.annotations:
            for ann in self.annotations[path]:
                x1, y1, x2, y2 = ann["bbox"]
                angle = ann.get("angle", 0.0)
                angle_str = f" ∠{angle:.1f}°" if angle != 0.0 else ""
                self.box_listbox.insert(
                    tk.END, f"[{ann['class']}]{angle_str} {x1:.0f},{y1:.0f} → {x2:.0f},{y2:.0f}"
                )

    def _delete_selected_box(self):
        path = self._current_path()
        if not path or path not in self.annotations:
            return
        sel = self.box_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.annotations[path].pop(idx)
        self._update_box_listbox()
        self._refresh_canvas()

    def clear_boxes(self):
        path = self._current_path()
        if path and path in self.annotations:
            self.annotations[path] = []
            self._update_box_listbox()
            self._refresh_canvas()

    def _add_class(self):
        name = simpledialog.askstring("Add Class", "Enter class name:", parent=self)
        if name and name not in self.class_list:
            self.class_list.append(name)
            self.class_combo["values"] = self.class_list
            self.class_combo.set(name)

    def _remove_class(self):
        cls = self.class_combo.get()
        if cls == DEFAULT_CLASS:
            messagebox.showwarning("Cannot remove", f"'{DEFAULT_CLASS}' is the default class.")
            return
        if cls in self.class_list:
            self.class_list.remove(cls)
            self.class_combo["values"] = self.class_list
            self.class_combo.set(self.class_list[0] if self.class_list else "")

    def save_annotations(self):
        """Save all YOLO .txt annotation files next to the source images."""
        if not self.annotations:
            messagebox.showinfo("Nothing to save", "No annotations to save.")
            return
        saved = 0
        for path, anns in self.annotations.items():
            if not anns:
                continue
            try:
                # use display dimensions (may differ from original if rotation was applied)
                if path in self.annotation_img_sizes:
                    iw, ih = self.annotation_img_sizes[path]
                else:
                    img = Image.open(path)
                    iw, ih = img.size
            except Exception:
                continue
            txt_path = os.path.splitext(path)[0] + ".txt"
            with open(txt_path, "w") as f:
                for ann in anns:
                    cls_idx = self.class_list.index(ann["class"]) if ann["class"] in self.class_list else 0
                    angle = ann.get("angle", 0.0)
                    if angle != 0.0:
                        corners = bbox_to_obb_corners(*ann["bbox"], angle)
                        coords = obb_corners_to_yolo(corners, iw, ih)
                        f.write(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in coords) + "\n")
                    else:
                        cx, cy, bw, bh = bbox_to_yolo(*ann["bbox"], iw, ih)
                        f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            saved += 1
        # save class names file alongside first image
        if self.image_paths:
            class_file = os.path.join(os.path.dirname(self.image_paths[0]), "classes.txt")
            with open(class_file, "w") as f:
                f.write("\n".join(self.class_list))
        messagebox.showinfo("Saved", f"Saved annotations for {saved} image(s).\nClasses: {class_file}")

    def _load_annotations_for_current(self):
        path = self._current_path()
        if not path:
            return
        txt_path = os.path.splitext(path)[0] + ".txt"
        if not os.path.exists(txt_path):
            messagebox.showinfo("Not found", f"No annotation file:\n{txt_path}")
            return
        try:
            img = Image.open(path)
            iw, ih = img.size
            anns = []
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    try:
                        if len(parts) == 9:
                            # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised)
                            cls_idx = int(parts[0])
                            coords = list(map(float, parts[1:]))
                            corners = [(coords[i] * iw, coords[i + 1] * ih) for i in range(0, 8, 2)]
                            x1, y1, x2, y2, angle_deg = corners_to_bbox_angle(corners)
                            cls = self.class_list[cls_idx] if cls_idx < len(self.class_list) else str(cls_idx)
                            ann = {"class": cls, "bbox": [x1, y1, x2, y2]}
                            if abs(angle_deg) > ANGLE_EPSILON:
                                ann["angle"] = angle_deg
                            anns.append(ann)
                        elif len(parts) == 5:
                            cls_idx, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                            cls = self.class_list[cls_idx] if cls_idx < len(self.class_list) else str(cls_idx)
                            x1, y1, x2, y2 = yolo_to_bbox(cx, cy, bw, bh, iw, ih)
                            anns.append({"class": cls, "bbox": [x1, y1, x2, y2]})
                    except (ValueError, IndexError):
                        continue  # skip malformed lines
            self.annotations[path] = anns
            self._update_box_listbox()
            self._refresh_canvas()
            self.status_var.set(f"Loaded {len(anns)} annotation(s) from {txt_path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load annotations:\n{exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────────
    def _choose_output_dir(self):
        d = filedialog.askdirectory(title="Choose Output Directory")
        if d:
            self.output_dir = d
            self.output_dir_lbl.config(text=d)

    def _get_target_size(self):
        mode = self.resize_mode.get()
        if mode == "none":
            return None
        if mode == "custom":
            return self.custom_w.get(), self.custom_h.get()
        # "WxH" string
        w, h = mode.split("x")
        return int(w), int(h)

    def batch_export(self):
        if not self.image_paths:
            messagebox.showwarning("No images", "Please load images first.")
            return
        if not self.output_dir:
            self._choose_output_dir()
        if not self.output_dir:
            return

        target_size = self._get_target_size()
        mode = self.export_mode.get()
        use_split = self.split_var.get() and mode in ("yolo", "yolo_obb")

        if use_split:
            # Validate split ratios
            train_pct = self.split_train.get()
            val_pct = self.split_val.get()
            test_pct = self.split_test.get()
            if train_pct + val_pct + test_pct != 100:
                messagebox.showerror(
                    "Invalid split",
                    f"Train + Val + Test must sum to 100 (currently {train_pct + val_pct + test_pct}).",
                )
                return

            # Shuffle and split image paths
            shuffled = list(self.image_paths)
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_pct / 100)
            n_val = int(n * val_pct / 100)
            split_map: dict[str, str] = {}
            for p in shuffled[:n_train]:
                split_map[p] = "train"
            for p in shuffled[n_train:n_train + n_val]:
                split_map[p] = "val"
            for p in shuffled[n_train + n_val:]:
                split_map[p] = "test"

            for split in ("train", "val", "test"):
                os.makedirs(os.path.join(self.output_dir, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir, split, "labels"), exist_ok=True)
        else:
            split_map = {}
            if mode in ("yolo", "yolo_obb"):
                img_dir = os.path.join(self.output_dir, "images")
                lbl_dir = os.path.join(self.output_dir, "labels")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)
            else:
                img_dir = os.path.join(self.output_dir, "images")
                os.makedirs(img_dir, exist_ok=True)

        exported = 0
        errors = []

        for path in self.image_paths:
            try:
                img = Image.open(path).convert("RGB")
                iw_orig, ih_orig = img.size

                # apply current sliders to every image
                img = ImageEnhance.Brightness(img).enhance(self.brightness_var.get())
                img = ImageEnhance.Contrast(img).enhance(self.contrast_var.get())
                img = ImageEnhance.Sharpness(img).enhance(self.sharpness_var.get())
                if self.grayscale_var.get():
                    img = ImageOps.grayscale(img).convert("RGB")
                if self.flip_h_var.get():
                    img = ImageOps.mirror(img)
                if self.flip_v_var.get():
                    img = ImageOps.flip(img)
                deg = self.rotate_var.get()
                if deg:
                    img = img.rotate(-deg, expand=True)

                if target_size:
                    img = img.resize(target_size, Image.LANCZOS)

                base = os.path.basename(path)
                name, _ = os.path.splitext(base)

                # Resolve destination directories
                if use_split:
                    split_name = split_map.get(path, "train")
                    cur_img_dir = os.path.join(self.output_dir, split_name, "images")
                    cur_lbl_dir = os.path.join(self.output_dir, split_name, "labels")
                else:
                    cur_img_dir = img_dir
                    cur_lbl_dir = lbl_dir if mode in ("yolo", "yolo_obb") else img_dir

                out_img = os.path.join(cur_img_dir, name + ".jpg")
                img.save(out_img, "JPEG", quality=95)

                # write YOLO / YOLO OBB labels
                if mode in ("yolo", "yolo_obb") and path in self.annotations and self.annotations[path]:
                    iw_new, ih_new = img.size
                    out_txt = os.path.join(cur_lbl_dir, name + ".txt")
                    ann_w, ann_h = self.annotation_img_sizes.get(path, (iw_orig, ih_orig))
                    scale_x = iw_new / ann_w
                    scale_y = ih_new / ann_h
                    with open(out_txt, "w") as f:
                        for ann in self.annotations[path]:
                            x1, y1, x2, y2 = ann["bbox"]
                            x1 *= scale_x
                            x2 *= scale_x
                            y1 *= scale_y
                            y2 *= scale_y
                            cls_idx = (
                                self.class_list.index(ann["class"])
                                if ann["class"] in self.class_list
                                else 0
                            )
                            angle = ann.get("angle", 0.0)
                            if mode == "yolo_obb":
                                corners = bbox_to_obb_corners(x1, y1, x2, y2, angle)
                                coords = obb_corners_to_yolo(corners, iw_new, ih_new)
                                f.write(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in coords) + "\n")
                            else:
                                if angle != 0.0:
                                    corners = bbox_to_obb_corners(x1, y1, x2, y2, angle)
                                    coords = obb_corners_to_yolo(corners, iw_new, ih_new)
                                    f.write(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in coords) + "\n")
                                else:
                                    cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, iw_new, ih_new)
                                    f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                exported += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        # write classes.txt and dataset.yaml for YOLO modes
        if mode in ("yolo", "yolo_obb"):
            with open(os.path.join(self.output_dir, "classes.txt"), "w") as f:
                f.write("\n".join(self.class_list))

            # Generate dataset.yaml (YOLOv8 format)
            if use_split:
                train_rel = "train/images"
                val_rel = "val/images"
                test_rel = "test/images"
            else:
                train_rel = "images"
                val_rel = "images"
                test_rel = "images"

            yaml_lines = [
                f"path: {os.path.abspath(self.output_dir)}",
                f"train: {train_rel}",
                f"val: {val_rel}",
                f"test: {test_rel}",
                f"nc: {len(self.class_list)}",
                f"names: {self.class_list}",
            ]
            yaml_path = os.path.join(self.output_dir, "dataset.yaml")
            with open(yaml_path, "w") as f:
                f.write("\n".join(yaml_lines) + "\n")

        msg = f"Exported {exported} / {len(self.image_paths)} image(s) to:\n{self.output_dir}"
        if use_split:
            msg += f"\n\nSplit: train={self.split_train.get()}%  val={self.split_val.get()}%  test={self.split_test.get()}%"
        if mode in ("yolo", "yolo_obb"):
            msg += f"\n\ndataset.yaml written."
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
        messagebox.showinfo("Export complete", msg)
        self.status_var.set(f"Batch export done → {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Auto-annotation
    # ──────────────────────────────────────────────────────────────────────────
    def _load_yolo_model(self, model_path: str):
        """Load and return a YOLO model; raises ImportError if ultralytics missing."""
        try:
            from ultralytics import YOLO  # lazy import
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "ultralytics is not installed.\nRun:  pip install ultralytics",
            )
            raise
        return YOLO(model_path)

    def auto_annotate(self):
        """Run a pre-trained YOLO model on the current image and add detections."""
        path = self._current_path()
        if path is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        model_path = filedialog.askopenfilename(
            title="Select YOLO Model (.pt)",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
        )
        if not model_path:
            return

        try:
            model = self._load_yolo_model(model_path)
        except ImportError:
            return
        except Exception as exc:
            messagebox.showerror("Model Error", f"Failed to load model:\n{exc}")
            return

        try:
            results = model(path)
            if path not in self.annotations:
                self.annotations[path] = []

            img = Image.open(path)
            iw, ih = img.size
            self.annotation_img_sizes[path] = (iw, ih)

            count = 0
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    self.annotations[path].append({"class": cls_name, "bbox": [x1, y1, x2, y2]})
                    if cls_name not in self.class_list:
                        self.class_list.append(cls_name)
                    count += 1

            self.class_combo["values"] = self.class_list
            self._update_box_listbox()
            self._refresh_canvas()
            self.status_var.set(f"Auto-annotated: {count} detection(s) on current image.")
        except Exception as exc:
            messagebox.showerror("Auto-Annotate Error", f"Inference failed:\n{exc}")

    def auto_annotate_all(self):
        """Run a pre-trained YOLO model on ALL loaded images (batch mode)."""
        if not self.image_paths:
            messagebox.showwarning("No images", "Please load images first.")
            return

        model_path = filedialog.askopenfilename(
            title="Select YOLO Model (.pt)",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
        )
        if not model_path:
            return

        try:
            model = self._load_yolo_model(model_path)
        except ImportError:
            return
        except Exception as exc:
            messagebox.showerror("Model Error", f"Failed to load model:\n{exc}")
            return

        total = len(self.image_paths)
        total_detections = 0
        errors = []

        for i, path in enumerate(self.image_paths):
            self.status_var.set(f"Auto-annotating {i + 1}/{total}: {os.path.basename(path)} …")
            self.update_idletasks()
            try:
                results = model(path)
                if path not in self.annotations:
                    self.annotations[path] = []

                img = Image.open(path)
                iw, ih = img.size
                self.annotation_img_sizes[path] = (iw, ih)

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        self.annotations[path].append({"class": cls_name, "bbox": [x1, y1, x2, y2]})
                        if cls_name not in self.class_list:
                            self.class_list.append(cls_name)
                        total_detections += 1
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self.class_combo["values"] = self.class_list
        self._update_box_listbox()
        self._refresh_canvas()

        msg = f"Batch auto-annotation complete.\n{total} image(s) processed, {total_detections} detection(s) added."
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
        self.status_var.set(f"Auto-annotated {total} image(s) — {total_detections} total detection(s).")
        messagebox.showinfo("Auto-Annotate All", msg)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ImageEditorApp()
    app.mainloop()
