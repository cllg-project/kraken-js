# kraken-js

JavaScript runtime for [Kraken](https://github.com/mittagessen/kraken) OCR/HTR models, targeting Node.js and Electron. Runs recognition and segmentation models exported from Kraken without a Python dependency at inference time.

## Features

- **`KrakenRecognizer`** — transcribe a line image to text with per-character confidence and position (VGSL and PP-OCRv6 models)
- **`KrakenSegmenter`** — locate text lines on a full page as oriented bounding boxes
- **`KrakenPipeline`** — full end-to-end pipeline: segment a page, deskew line crops, recognize text
- Single-file model format (`.js_mlmodel`) bundles the ONNX graph and metadata
- Hardware acceleration via ONNX Runtime execution providers (CoreML, DirectML, CUDA, WebGPU)
- Batch inference for recognition

## Installation

**As a dependency in your project:**

```bash
npm install github:cllg-project/kraken-js
```

```js
const { KrakenRecognizer, KrakenSegmenter, KrakenPipeline } = require('kraken-js');
```

Requires Node.js ≥ 18. The `sharp` and `onnxruntime-node` native binaries are installed automatically.

**To develop or run tests in this repo:**

```bash
npm install
```

## Model format: `.js_mlmodel`

A `.js_mlmodel` is a ZIP archive containing:

```
model.onnx       ONNX graph (dynamic batch + width axes)
metadata.json    model configuration (see below)
```

Export a Kraken `.mlmodel` or `.safetensors` model with the provided Python script (requires a Kraken Python environment):

```bash
# Install Kraken into a venv (once). onnx is required to write the graph,
# onnxruntime to run the post-export parity check.
python3 -m venv env && env/bin/pip install kraken onnx onnxruntime

# Recognition model (VGSL or PP-OCRv6)
env/bin/python3 export_kraken_onnx.py model_best.mlmodel
# → model_best.js_mlmodel

# Segmentation model
env/bin/python3 export_kraken_onnx.py segmentation.mlmodel
# → segmentation.js_mlmodel

# Custom output path
env/bin/python3 export_kraken_onnx.py model.mlmodel /path/to/output.js_mlmodel

# Skip the parity check (not recommended)
env/bin/python3 export_kraken_onnx.py model.mlmodel --no-check
```

After writing the graph the exporter re-runs it under ONNX Runtime at several
widths and compares against the original torch model. This is not a formality:
the TorchScript tracer silently freezes shape-derived values into the graph, and
without the check a model can export cleanly and then be wrong at every width but
the one it was traced at.

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

### PP-OCRv6 recognition metadata

Kraken ≥ 7.1 `PPOCRv6Model` recognizers (PPLCNetV4 backbone → LightSVTR neck → CTC head)
export to the same container, with two extra graph tensors:

```json
{
  "model_type": "recognition",
  "architecture": "ppocrv6",
  "variant": "medium",
  "height": 96,
  "channels": 3,
  "pad": 16,
  "one_channel_mode": "RGB",
  "num_classes": 1630,
  "width_subsampling": 8,
  "seq_lens_input": true,
  "vgsl": "",
  "codec": { " ": [1], "a": [2] }
}
```

`seq_lens_input` means the graph takes a second input, `seq_lens` (int64, one
unpadded width per batch element), and returns a second output, `out_lens` (the
valid output width per batch element). The neck mixes across the whole sequence
with global attention, so the batch padding has to be masked out — `KrakenRecognizer`
feeds and consumes both automatically, and preprocessing is unchanged from VGSL
models. Everything else (`recognize`, `recognizeBatch`, `KrakenPipeline`) works
the same way.

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

> **Note on batching**: `recognizeBatch` pads shorter images to the widest in the batch. PP-OCRv6 models mask that padding out of their attention neck via `seq_lens`, so batching is safe; results are not bit-identical to single-line inference, because the backbone's SAME padding phase depends on the batch width (Kraken's own torch model behaves the same way). For VGSL models the padded frames are blank and the backward LSTM pass sees them, so grouping lines of similar width still gives the most faithful results.

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

#### Reading order and double-page spreads

Lines are returned sorted top-to-bottom, left-to-right by default. When a landscape image (width > height × 1.2) is detected, `KrakenSegmenter` looks for a vertical gutter in the centre of the page — a horizontal gap in the distribution of line centres. If one is found, lines are split into left and right columns, each sorted independently by their vertical position, and concatenated (left column first).

To disable this heuristic — for example when a wide single page is mistakenly split, or for right-to-left scripts — pass `noColumnSplit: true`:

```js
const seg = await KrakenSegmenter.create('./segmentation.js_mlmodel', {
  noColumnSplit: true,
});
```

Other segmenter options:

| Option | Default | Description |
|--------|---------|-------------|
| `threshold` | `0.5` | Sigmoid threshold for baseline heatmap binarisation |
| `minArea` | `20` | Minimum connected-component area in heatmap pixels |
| `noColumnSplit` | `false` | Disable double-page column detection |
| `executionProviders` | `['cpu']` | ONNX Runtime execution providers |
| `threads` | one per physical core | Threads per inference (see [Limiting CPU usage](#limiting-cpu-usage)) |
| `allowSpinning` | ONNX Runtime default (`true`) | `false` stops idle threads from busy-waiting between runs |
| `sessionOptions` | `{}` | Raw ONNX Runtime session options (override the two above) |

`KrakenRecognizer.create` accepts the same `executionProviders`, `threads`, `allowSpinning` and `sessionOptions` options.

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

Pipeline options (passed as the third argument to `KrakenPipeline.create`):

| Option | Default | Description |
|--------|---------|-------------|
| `expandUp` | `0.85` | Fraction of estimated line height to include above the baseline |
| `expandDown` | `0.35` | Fraction of estimated line height to include below the baseline |
| `threads` | one per physical core | Threads per inference, for both models |
| `allowSpinning` | ONNX Runtime default (`true`) | `false` stops idle threads of both models from busy-waiting |
| `sharpConcurrency` | sharp default (one per core) | Cap libvips threads. **Process-wide**: calls `sharp.concurrency(n)` |
| `segmenter` | `{}` | Options forwarded to `KrakenSegmenter.create` (override the pipeline-level values) |
| `recognizer` | `{}` | Options forwarded to `KrakenRecognizer.create` (override the pipeline-level values) |

```js
const pipeline = await KrakenPipeline.create(segPath, recPath, {
  expandDown: 0.5,  // more room for descenders
});
```

### Limiting CPU usage

By default ONNX Runtime gives each session one thread per physical core, and idle
threads busy-wait between inferences, so a running pipeline keeps every core at 100%.
For a desktop application that is rarely worth it:

```js
const pipeline = await KrakenPipeline.create(segPath, recPath, {
  threads: 4,            // per inference, for both models
  allowSpinning: false,  // let idle threads sleep
});
```

Measured on `tests/fixtures/fullpage.png` (35 lines, 24 logical cores, single runs):

| Options | Wall time | Average cores busy | Threads |
|---|---|---|---|
| defaults | 4.4 s | 20.0 | 59 |
| `allowSpinning: false` | 4.2 s | 7.3 | 59 |
| `threads: 4` | 4.5 s | 3.9 | 19 |
| `threads: 4, allowSpinning: false` | 5.0 s | 3.0 | 19 |
| `threads: 2, allowSpinning: false` | 7.0 s | 2.0 | 15 |

Note that `onnxruntime-node` runs each inference synchronously on the JavaScript thread
(after a `setImmediate`), so inferences never overlap and block the event loop while
they run; in Electron, run the pipeline away from the UI's process if that matters.

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
| Color mode | grayscale (L) or RGB per `channels` (PP-OCRv6: always RGB) | RGB (3-channel) |
| Resize | height = model `height`, proportional width | height = model `height`, proportional width |
| Padding | `pad` px white on each side | none |
| Normalize | `1 − x/255` | `1 − x/255` |
| Layout | CHW Float32 | CHW Float32 |

## Known limitations

### No mask support

Python Kraken's `segment()` accepts a binary mask image to exclude regions (margins, illustrations, decorations) from segmentation. `KrakenSegmenter.segment()` has no equivalent — every pixel in the image is processed unconditionally. To work around this, crop or blank out regions in the image before passing it to the segmenter.

### Straight-line OBBs only — no polyline baselines

Python Kraken represents each detected text line as a **polyline baseline** (a sequence of (x, y) points that follow the actual path of the text) plus a bounding polygon derived from that polyline. This handles curved, wavy, or heavily skewed lines accurately.

`KrakenSegmenter` instead returns a single **oriented bounding rectangle** (OBB) per line, computed via PCA on the connected-component pixels in the heatmap. This works well for straight or gently diagonal lines but:

- Curved or wavy baselines are approximated as a single best-fit rectangle; the approximation degrades as curvature increases.
- `corners` is always a 4-point rectangle, not a line-hugging polygon.
- Very long or highly skewed lines may produce a poor bounding box if the PCA axis does not align with the true reading direction.

## Running tests

```bash
npm test               # full suite
npm run test:smoke     # quick end-to-end smoke test on example_line.png
```

The PP-OCRv6 tests need a ~57 MB model fixture that is not committed; they skip
until you build it (needs the Kraken venv described above):

```bash
npm run fixtures:ppocr   # downloads the model from Zenodo and exports it
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
    double_page.png             landscape double-page spread (1722×1435)
    example.txt                 ground-truth transcription of fullpage.png
    *.xml                       ALTO XML ground-truth segmentation

docs/
  index.html      browser demo (GitHub Pages)
  demo.js         browser pipeline — onnxruntime-web + Canvas API, no server required

debug_segmentation.js   CLI tool: overlay OBBs and crop polygons on an image for debugging
export_kraken_onnx.py   Python export script (requires a Kraken-capable venv)
```

## Demo

An interactive browser demo is available at **https://cllg-project.github.io/kraken-js/**.

Load the bundled sample page or drag-and-drop your own image, then click **Run OCR**. The full segmentation + recognition pipeline runs entirely client-side via WebAssembly — no server, no data upload.

Two recognition models are available via the model selector:
- **Greek print (Antiqua Graeca)** — Ancient Greek polytonic print
- **Latin script print (CATMuS)** — Latin-script print ([CATMuS large](https://zenodo.org/records/10592716))

The segmentation model is shared across both. A **Toggle overlay** button draws the oriented crop polygons (the exact regions sent to the recognizer) over the page image. Double-page spreads are split into columns automatically; a checkbox disables this if needed.

### Keeping the page responsive

Inference is heavy enough to freeze a page if it runs on the main thread — a full-page
segmentation pass is seconds of uninterruptible work. The demo avoids that:

- `ort.env.wasm.proxy = true` puts ONNX Runtime's model compilation and every `run()`
  in its own worker. Measured on the sample page: 5.5 s of blocked main thread and a
  3.8 s single freeze without it, versus 0.1 s blocked and a 52 ms longest task with it.
- Sessions are cached per model URL, so a second run — or switching recognition models
  and back — costs no download and no recompilation.
- Model downloads report progress through the status line.

`npm run demo` additionally serves `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`,
which is what lets ONNX Runtime use multi-threaded WASM (6 s vs 17 s for the sample page).
GitHub Pages cannot send those headers, so the deployed demo is single-threaded — slower,
but still off the main thread and responsive. Because those headers apply locally, the
jsdelivr `<script>` tags in `docs/index.html` must carry `crossorigin="anonymous"`.

## Acknowledgements

This JavaScript runtime is built on top of [Kraken](https://github.com/mittagessen/kraken), the OCR/HTR engine created and maintained by [Benjamin Kiessling](https://github.com/mittagessen).

The project *Corpus Liberatum Linguae Graecae* was supported by the French National Research Agency (ANR) under the France 2030 grant reference number « ANR-24-RRII-0002 » operated by the Inria Quadrant Program.

Project Leader: Thibault Clérice.
Project Members: Nicolas Angleraud, Antonia Karamolegkou, Benoît Sagot.

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2017 Benjamin Kiessling.
