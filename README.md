# Number Plate Image Editor – YOLO & CNN

A Python desktop GUI application for preparing and annotating images for **YOLO** object detection and **CNN** number plate recognition models.

---

## Features

| Feature | Description |
|---|---|
| 📂 Open images | Select individual files or an entire folder |
| 🔲 YOLO annotation | Draw bounding boxes with class labels; auto-save `.txt` label files |
| 🔷 OBB annotation | Draw oriented (rotated) bounding boxes; export in YOLO OBB format |
| 🎨 Image transforms | Brightness, contrast, sharpness, grayscale, flip, rotate |
| 📐 Resize presets | YOLOv3/v4 (416×416), YOLOv5/v8 (640×640), ResNet/VGG (224×224), and more |
| 📤 Batch export | Process and export all images at once in YOLO, YOLO OBB, or CNN format |
| 💾 Load annotations | Re-load existing YOLO / YOLO OBB `.txt` label files for editing |

---

## Requirements

- Python 3.10+
- [Pillow](https://pypi.org/project/Pillow/) (image processing)
- `tkinter` (bundled with most Python installations)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python image_editor.py
```

### Workflow

1. **Open Images** – click *📂 Open Images* or *📁 Open Folder*.
2. **Annotate** – draw bounding boxes directly on the canvas; assign class labels in the *Annotate* tab.
3. **OBB** – tick *OBB Mode* in the *Annotate* tab, set an angle, then draw a box to create an oriented bounding box.
4. **Adjust** – use the *Transforms* tab to apply brightness/contrast/flip/rotate for data augmentation.
5. **Export** – configure output format and resize preset in the *Export* tab, then click **Batch Export All**.

### OBB (Oriented Bounding Box)

Enable **OBB Mode** in the *Annotate* tab and enter a rotation **Angle (°)** before drawing a box. The box will be stored with that rotation and rendered as a rotated polygon on the canvas. Angles are in degrees (positive = clockwise).

OBB annotations are saved/loaded in **YOLO OBB format** (9 values per line):

```
<class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>   ← normalised corner coords
```

For batch export, select **YOLO OBB** in the *Export* tab to write all annotations (including axis-aligned boxes) in the 9-value OBB format required by YOLOv8 OBB models.

### YOLO output structure

```
output/
├── images/
│   ├── img001.jpg
│   └── ...
├── labels/
│   ├── img001.txt   ← YOLO format: <class> <cx> <cy> <w> <h> (normalised)
│   └── ...
└── classes.txt
```

### YOLO OBB output structure

```
output/
├── images/
│   ├── img001.jpg
│   └── ...
├── labels/
│   ├── img001.txt   ← YOLO OBB format: <class> <x1> <y1> … <x4> <y4> (normalised)
│   └── ...
└── classes.txt
```

### CNN output structure

```
output/
└── images/
    ├── img001.jpg
    └── ...
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| `←` / `→` | Previous / next image |
| `Delete` | Delete selected bounding box |