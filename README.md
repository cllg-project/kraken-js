# kraken-js

JavaScript runtime for [Kraken](https://github.com/mittagessen/kraken) OCR/HTR models, targeting Node.js and Electron. Runs recognition and segmentation models exported from Kraken without a Python dependency at inference time.

## Features

- **`KrakenRecognizer`** — transcribe a line image to text with per-character confidence and position
- **`KrakenSegmenter`** — locate text lines on a full page as oriented bounding boxes
- **`KrakenPipeline`** — full end-to-end pipeline: segment a page, deskew line crops, recognize text
- Single-file model format (`.js_mlmodel`) bundles the ONNX graph and metadata
- Hardware acceleration via ONNX Runtime execution providers (CoreML, DirectML, CUDA, WebGPU)
- Batch inference for recognition

## Installation

```bash
npm install
```

Requires Node.js ≥ 18. The `sharp` and `onnxruntime-node` native binaries are installed automatically via npm.

## Model format: `.js_mlmodel`

A `.js_mlmodel` is a ZIP archive containing:

```
model.onnx       ONNX graph (dynamic batch + width axes)
metadata.json    model configuration (see below)
```

Export a Kraken `.mlmodel` or `.safetensors` model with the provided Python script:

```bash
# Install Kraken into the project venv (once)
env/bin/pip install -e "."

# Recognition model
env/bin/python3 export_kraken_onnx.py model_best.mlmodel
# → model_best.js_mlmodel

# Segmentation model
env/bin/python3 export_kraken_onnx.py segmentation.mlmodel
# → segmentation.js_mlmodel
```

### Recognition metadata

```json
{
  "model_type": "recognition",
  "height": 120,
  "channels": 1,
  "pad": 16,
  "one_channel_mode": "L",
  "vgsl": "[1,120,0,1 Cr4,2,32 ... O1c232]",
  "codec": { " ": [1], "a": [2], "æ": [4, 5] }
}
```

### Segmentation metadata

```json
{
  "model_type": "segmentation",
  "height": 1800,
  "channels": 3,
  "class_mapping": {
    "aux": { "_start_separator": 0, "_end_separator": 1 },
    "baselines": { "DefaultLine-Margin": 2, "DefaultLine": 3 }
  },
  "topline": false,
  "vgsl": "[1,1800,0,3 Cr7,7,64 ... O2l4]"
}
```

## Usage

### Recognition

```js
const { KrakenRecognizer } = require('./src');

const r = await KrakenRecognizer.create('./model_best.js_mlmodel');

// Single image (path or Buffer)
const { text, chars } = await r.recognize('./line.png');
// text: "ζετε, ἀλλὰ ἀλόγῳ πάθει..."
// chars: [{ char, conf, x0, x1 }, ...]  — x0/x1 in original image pixels

// Batch (single ONNX forward pass — use same-height images)
const results = await r.recognizeBatch(['line1.png', 'line2.png']);
```

`chars` positions are scaled back to the original (pre-resize, pre-pad) image width.

> **Note on batching**: mixing images of very different widths in `recognizeBatch` can corrupt results for the shorter images due to zero-padding interacting with the backward LSTM pass. Prefer `recognize` for variable-length lines, or group lines of similar width.

### Segmentation

```js
const { KrakenSegmenter } = require('./src');

const seg = await KrakenSegmenter.create('./segmentation.js_mlmodel');

const { lines, imageSize } = await seg.segment('./page.png');
// lines: [{ obb, type }, ...]  — sorted top-to-bottom, left-to-right
// imageSize: { width, height }  — original image dimensions
```

Each line object:

```js
{
  obb: {
    cx, cy,        // baseline centre in original image pixels
    w, h,          // OBB dimensions (w = along text direction, h = baseline width)
    angle,         // radians of text direction from +x, in (-π/2, π/2]
    corners,       // [[x,y], [x,y], [x,y], [x,y]] clockwise from top-left
  },
  type: 'DefaultLine' | 'DefaultLine-Margin'
}
```

> **OBB height note**: the model predicts thin baselines (~1–2 px in heatmap space), so `obb.h` reflects the baseline width, not the full text height. `KrakenPipeline` derives the crop height from inter-line spacing automatically.

### Full pipeline

```js
const { KrakenPipeline } = require('./src');

const pipeline = await KrakenPipeline.create(
  './segmentation.js_mlmodel',
  './model_best.js_mlmodel'
);

const lines = await pipeline.process('./page.png');
// lines: [{ obb, type, text, chars }, ...]  — reading order
```

Each result:

```js
{
  obb:   { cx, cy, w, h, angle, corners },  // in original image coords
  type:  'DefaultLine' | 'DefaultLine-Margin',
  text:  'τοῦ κηρύγματος αὐτῆς, ...',
  chars: [{ char, conf, x0, x1 }, ...]
}
```

The pipeline resizes each line crop to the recognizer's expected height, so the `chars` `x0`/`x1` coordinates are relative to the **crop**, not the full page.

### Hardware acceleration

```js
// macOS (Node.js)
const r = await KrakenRecognizer.create('./model.js_mlmodel', {
  executionProviders: ['coreml', 'cpu'],
});

// Windows
const r = await KrakenRecognizer.create('./model.js_mlmodel', {
  executionProviders: ['directml', 'cpu'],
});

// Linux with NVIDIA GPU
const r = await KrakenRecognizer.create('./model.js_mlmodel', {
  executionProviders: ['cuda', 'cpu'],
});

// Electron renderer / browser
const r = await KrakenRecognizer.create('./model.js_mlmodel', {
  executionProviders: ['webgpu', 'cpu'],
});
```

The same `executionProviders` option is accepted by `KrakenSegmenter.create` and `KrakenPipeline.create` (pass via `opts.segmenter` / `opts.recognizer`):

```js
const pipeline = await KrakenPipeline.create(segPath, recPath, {
  segmenter:  { executionProviders: ['coreml', 'cpu'] },
  recognizer: { executionProviders: ['coreml', 'cpu'] },
});
```

## Preprocessing

All models share the same normalization: pixels are divided by 255 then inverted (`value = 1 − pixel/255`), matching Kraken's `tensor_invert` transform.

| Step | Recognition | Segmentation |
|------|------------|--------------|
| Color mode | grayscale (L) or RGB per `channels` | RGB (3-channel) |
| Resize | height = model `height`, proportional width | height = model `height`, proportional width |
| Padding | `pad` px white on each side | none |
| Normalize | `1 − x/255` | `1 − x/255` |
| Layout | CHW Float32 | CHW Float32 |

## Running tests

```bash
npm test               # full suite (103 tests)
npm run test:smoke     # quick end-to-end smoke test on example_line.png
```

## Project layout

```
src/
  index.js        public exports (KrakenRecognizer, KrakenSegmenter, KrakenPipeline)
  recognizer.js   KrakenRecognizer — line image → text
  segmenter.js    KrakenSegmenter  — page image → oriented bounding boxes
  pipeline.js     KrakenPipeline   — segment + deskew + recognize
  preprocess.js   image → Float32Array (sharp)
  decode.js       greedy CTC decoder + codec lookup
  heatmap.js      segmentation post-processing (threshold, connected components, OBB via PCA)
  loader.js       .js_mlmodel ZIP reader

tests/
  recognizer.test.js
  segmenter.test.js
  pipeline.test.js
  preprocess.test.js
  decode.test.js
  loader.test.js
  fixtures/
    model_best.js_mlmodel       recognition model
    segmentation.js_mlmodel     segmentation model
    example_line.png            single line image
    ood_example.png             out-of-distribution line image
    fullpage.png                full manuscript page (2479×3508)
    example.txt                 ground-truth transcription of fullpage.png
    *.xml                       ALTO XML ground-truth segmentation

export_kraken_onnx.py   Python export script (run from the kraken repo root)
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2017 Benjamin Kiessling.
