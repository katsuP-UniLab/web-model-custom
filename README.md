# YOLO Web Object Detection (Custom Model & Labels)

Run YOLOv8 / YOLOv11 object detection directly in the browser using [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

This application supports loading custom ONNX models from any URL or local path, alongside dynamic label configuration via JSON URL fetching or direct textarea editing.

---

## Features

- **Custom Model URL / Upload**: Load ONNX models via remote URL, local path, or direct file upload (`.onnx`).
- **Dynamic Labels**:
  - **Fetch from URL**: Retrieve label mappings directly from a remote or local `labels.json`.
  - **Upload File**: Upload `.json` or `.txt` label files directly.
  - **Textarea Editor**: Paste or edit class names as a JSON array or JSON key-value map, with instant canvas re-rendering.
- **Client-Side Inference**: Runs entirely in the client browser using WebAssembly (`wasm`).
- **Interactive Canvas**: Real-time bounding box visualization with confidence scores.
- **Catppuccin Themes**: Styled with Catppuccin Macchiato (dark) and Latte (light), with dynamic theme toggle and auto-detection.

---

## Getting Started

Because browsers enforce security restrictions (such as CORS and `file://` fetch limitations), you must serve the files via a local HTTP server.

### Option 1: Using Python

```bash
# Python 3
python3 -m http.server 8000
```

Open [http://localhost:8000](http://localhost:8000) in your web browser.

### Option 2: Using Node.js

```bash
# Using npx serve
npx serve .

# Or using http-server
npx http-server -p 8000
```

Open the printed localhost URL in your browser.

---

## How to Use

### 1. Model Configuration
- **Load via URL or path**: Enter URL or path in input field (e.g., `best.onnx` or `https://example.com/best.onnx`) and click **Load Model**.
- **Upload directly**: Click **Choose File** under "Or upload .onnx file directly" to load a local `.onnx` model into memory without placing it in the server folder.

### 2. Labels Configuration
You can configure labels using three methods:

- **Option A (Fetch from URL)**: Enter the URL pointing to a JSON file (e.g., `labels.json` or `https://your-domain.com/labels.json`) and click **Fetch URL**. The fetched labels will automatically populate the textarea.
- **Option B (Upload File)**: Upload a local `.json` or `.txt` label file directly. The contents will load and sync into the textarea.
- **Option C (Textarea)**: Paste or type your labels directly into the textarea and click **Apply Textarea Labels**.

#### Supported Label Formats:
1. **JSON Array** (indexed by position `0, 1, 2...`):
   ```json
   [
     "french bulldog",
     "german shepherd",
     "golden retriever"
   ]
   ```
2. **JSON Object** (keyed by class ID):
   ```json
   {
     "0": "french bulldog",
     "1": "german shepherd",
     "2": "golden retriever"
   }
   ```
3. **Newline-separated text** (will automatically be converted into a JSON array):
   ```text
   french bulldog
   german shepherd
   golden retriever
   ```

*Tip: If an image is already detected on the canvas, updating labels and clicking **Apply Textarea Labels** will re-render bounding box names immediately.*

### 3. Run Detection
- Click **Choose File** under Section 3 and select an image (`.jpg`, `.png`, `.webp`, etc.).
- The image will be resized to 640×640, preprocessed into an RGB float32 tensor `[1, 3, 640, 640]`, passed to the ONNX session, and filtered via Non-Maximum Suppression (NMS).
- Detected bounding boxes and class predictions are rendered directly onto the canvas.

---

## Exporting YOLO to ONNX

To export your custom YOLO model for use with this web app:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Export to ONNX format with 640x640 input resolution
model.export(format="onnx", imgsz=640, opset=12)
```

This generates `best.onnx` with:
- Input shape: `[1, 3, 640, 640]`
- Output shape: `[1, 4 + num_classes, 8400]` (for standard YOLOv8 / YOLOv11 detection models).

---

## Remote Hosting & CORS Notice

When loading models or label files from an external server (e.g., AWS S3, Cloudflare R2, GitHub Pages, or a custom API), ensure that the hosting server sends appropriate CORS headers:

```http
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, OPTIONS
Access-Control-Allow-Headers: Range, Content-Type
```

Without CORS headers, the browser will block requests to external `.onnx` and `.json` files.

