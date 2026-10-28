/*
==================================================
 YOLOv11 Object Detection using ONNX Runtime Web
 Dynamic Model & Labels Loader
==================================================
*/

// UI Elements
const modelUrlInput = document.getElementById("modelUrl");
const loadModelBtn = document.getElementById("loadModelBtn");
const modelStatus = document.getElementById("modelStatus");

const labelUrlInput = document.getElementById("labelUrl");
const fetchLabelsBtn = document.getElementById("fetchLabelsBtn");
const labelsTextarea = document.getElementById("labelsTextarea");
const applyLabelsBtn = document.getElementById("applyLabelsBtn");
const labelStatus = document.getElementById("labelStatus");

const imageInput = document.getElementById("imageInput");
const inferenceStatus = document.getElementById("inferenceStatus");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Runtime State
let session = null;
let labels = [];
let lastImage = null;
let lastBoxes = [];

// ==================================================
// Status Helpers
// ==================================================
function setStatus(element, text, type = "normal") {
    element.textContent = text;
    element.className = `status-badge ${type}`;
}

// ==================================================
// 1. Model Loading
// ==================================================
async function loadModel(url) {
    if (!url || !url.trim()) {
        setStatus(modelStatus, "Please enter a valid model URL or path.", "error");
        return;
    }

    const trimmedUrl = url.trim();
    setStatus(modelStatus, `Loading model from ${trimmedUrl}...`, "loading");
    loadModelBtn.disabled = true;

    try {
        session = await ort.InferenceSession.create(trimmedUrl, {
            executionProviders: ["wasm"]
        });

        const inputName = session.inputNames[0] || "input";
        const outputName = session.outputNames[0] || "output";
        setStatus(modelStatus, `Model loaded successfully! (Input: ${inputName}, Output: ${outputName})`, "success");
        console.log("Model loaded successfully:", trimmedUrl, session);

        // If an image is already loaded, re-run inference
        if (lastImage) {
            runInference(lastImage);
        }
    } catch (err) {
        console.error("Error loading model:", err);
        setStatus(modelStatus, `Failed to load model: ${err.message}`, "error");
    } finally {
        loadModelBtn.disabled = false;
    }
}

loadModelBtn.addEventListener("click", () => {
    loadModel(modelUrlInput.value);
});

modelUrlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        loadModel(modelUrlInput.value);
    }
});

// ==================================================
// 2. Labels Management
// ==================================================
function parseLabelsFromText(rawText) {
    const text = rawText.trim();
    if (!text) {
        labels = [];
        setStatus(labelStatus, "No labels provided.", "error");
        return false;
    }

    try {
        // Attempt 1: Parse JSON
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
            labels = parsed;
            setStatus(labelStatus, `Loaded ${labels.length} classes from JSON array.`, "success");
            return true;
        } else if (typeof parsed === "object" && parsed !== null) {
            labels = parsed;
            const count = Object.keys(labels).length;
            setStatus(labelStatus, `Loaded ${count} classes from JSON object.`, "success");
            return true;
        }
    } catch (err) {
        // Attempt 2: Fallback to newline-separated list
        const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length > 0) {
            labels = lines;
            // Update textarea to valid JSON representation
            labelsTextarea.value = JSON.stringify(labels, null, 2);
            setStatus(labelStatus, `Parsed ${labels.length} classes (converted to JSON array).`, "success");
            return true;
        }
        setStatus(labelStatus, `Failed to parse labels: ${err.message}`, "error");
        return false;
    }

    return false;
}

async function fetchLabelsFromUrl(url) {
    if (!url || !url.trim()) {
        setStatus(labelStatus, "Please enter a valid labels URL.", "error");
        return;
    }

    const trimmedUrl = url.trim();
    setStatus(labelStatus, `Fetching labels from ${trimmedUrl}...`, "loading");
    fetchLabelsBtn.disabled = true;

    try {
        const response = await fetch(trimmedUrl);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        labels = data;
        labelsTextarea.value = JSON.stringify(data, null, 2);

        const count = Array.isArray(data) ? data.length : Object.keys(data).length;
        setStatus(labelStatus, `Loaded ${count} classes from URL.`, "success");

        // Re-draw if boxes exist
        if (lastImage && lastBoxes.length > 0) {
            redrawCanvas();
        }
    } catch (err) {
        console.error("Error fetching labels:", err);
        setStatus(labelStatus, `Failed to fetch labels: ${err.message}`, "error");
    } finally {
        fetchLabelsBtn.disabled = false;
    }
}

fetchLabelsBtn.addEventListener("click", () => {
    fetchLabelsFromUrl(labelUrlInput.value);
});

labelUrlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        fetchLabelsFromUrl(labelUrlInput.value);
    }
});

applyLabelsBtn.addEventListener("click", () => {
    const success = parseLabelsFromText(labelsTextarea.value);
    if (success && lastImage && lastBoxes.length > 0) {
        redrawCanvas();
    }
});

// ==================================================
// 3. Image Upload & Inference Pipeline
// ==================================================
imageInput.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        lastImage = img;
        runInference(img);
    };
    img.src = URL.createObjectURL(file);
});

async function runInference(img) {
    if (!session) {
        setStatus(inferenceStatus, "Please load an ONNX model before running inference.", "error");
        canvas.width = 640;
        canvas.height = 640;
        ctx.clearRect(0, 0, 640, 640);
        ctx.drawImage(img, 0, 0, 640, 640);
        return;
    }

    setStatus(inferenceStatus, "Running inference...", "loading");

    try {
        canvas.width = 640;
        canvas.height = 640;
        ctx.drawImage(img, 0, 0, 640, 640);

        // Preprocess
        const tensor = preprocessImage(img);

        // Run model
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;

        const startTime = performance.now();
        const result = await session.run(feeds);
        const inferenceTime = (performance.now() - startTime).toFixed(1);

        const outputTensor = result[session.outputNames[0]];

        // Postprocess & NMS
        let boxes = postprocessOutput(outputTensor, 0.35);
        boxes = nonMaxSuppression(boxes, 0.35);
        lastBoxes = boxes;

        // Render
        redrawCanvas();

        setStatus(inferenceStatus, `Detected ${boxes.length} objects in ${inferenceTime} ms.`, "success");
    } catch (err) {
        console.error("Inference error:", err);
        setStatus(inferenceStatus, `Inference error: ${err.message}`, "error");
    }
}

function redrawCanvas() {
    if (!lastImage) return;

    ctx.clearRect(0, 0, 640, 640);
    ctx.drawImage(lastImage, 0, 0, 640, 640);
    drawBoxes(lastBoxes);
}

// ==================================================
// 4. Preprocessing: Image -> Tensor [1, 3, 640, 640]
// ==================================================
function preprocessImage(img) {
    const size = 640;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = size;
    tempCanvas.height = size;

    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(img, 0, 0, size, size);

    const imageData = tempCtx.getImageData(0, 0, size, size);
    const data = imageData.data;

    const float32 = new Float32Array(3 * size * size);

    for (let i = 0; i < size * size; i++) {
        // Red Channel
        float32[i] = data[i * 4] / 255.0;
        // Green Channel
        float32[i + size * size] = data[i * 4 + 1] / 255.0;
        // Blue Channel
        float32[i + 2 * size * size] = data[i * 4 + 2] / 255.0;
    }

    return new ort.Tensor("float32", float32, [1, 3, size, size]);
}

// ==================================================
// 5. Postprocessing: Output -> Bounding Boxes
// ==================================================
function postprocessOutput(output, threshold = 0.35) {
    const data = output.data;
    const dims = output.dims; // e.g. [1, numAttrs, numBoxes]

    const numAttrs = dims[1];
    const numBoxes = dims[2];
    const boxes = [];

    for (let i = 0; i < numBoxes; i++) {
        const x = data[i];
        const y = data[numBoxes + i];
        const w = data[numBoxes * 2 + i];
        const h = data[numBoxes * 3 + i];

        let bestScore = 0;
        let classId = 0;

        for (let c = 4; c < numAttrs; c++) {
            const score = data[numBoxes * c + i];
            if (score > bestScore) {
                bestScore = score;
                classId = c - 4;
            }
        }

        if (bestScore < threshold) continue;

        boxes.push({
            x: x - w / 2,
            y: y - h / 2,
            width: w,
            height: h,
            score: bestScore,
            classId: classId
        });
    }

    return boxes;
}

// ==================================================
// 6. Non-Max Suppression (NMS)
// ==================================================
function nonMaxSuppression(boxes, iouThreshold = 0.35) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];

    while (boxes.length) {
        const current = boxes.shift();
        selected.push(current);

        boxes = boxes.filter(box => {
            if (box.classId !== current.classId) {
                return true;
            }
            return calculateIoU(current, box) < iouThreshold;
        });
    }

    return selected;
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = a.width * a.height;
    const areaB = b.width * b.height;

    return intersection / (areaA + areaB - intersection);
}

// ==================================================
// 7. Draw Bounding Boxes
// ==================================================
function drawBoxes(boxes) {
    ctx.lineWidth = 3;
    ctx.font = "16px sans-serif";

    boxes.forEach(box => {
        // Determine label text
        let className = `Class ${box.classId}`;
        if (Array.isArray(labels) && labels[box.classId] !== undefined) {
            className = labels[box.classId];
        } else if (typeof labels === "object" && labels !== null && labels[box.classId] !== undefined) {
            className = labels[box.classId];
        }

        const scorePercent = (box.score * 100).toFixed(1);
        const caption = `${className} (${scorePercent}%)`;

        // Box border
        ctx.strokeStyle = "#22c55e";
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // Label tag background
        ctx.font = "bold 14px sans-serif";
        const textMetrics = ctx.measureText(caption);
        const textHeight = 18;
        const tagY = Math.max(0, box.y - textHeight - 4);

        ctx.fillStyle = "#22c55e";
        ctx.fillRect(box.x, tagY, textMetrics.width + 8, textHeight + 4);

        // Label text
        ctx.fillStyle = "#000000";
        ctx.fillText(caption, box.x + 4, tagY + textHeight - 1);
    });
}

// ==================================================
// Initial Auto-Load
// ==================================================
window.addEventListener("DOMContentLoaded", () => {
    // Attempt to load default labels
    fetchLabelsFromUrl(labelUrlInput.value);

    // Attempt to load default model
    loadModel(modelUrlInput.value);
});
