/**
 * Browser-side kraken-js pipeline.
 *
 * Uses:
 *   - onnxruntime-web (ESM) for inference
 *   - JSZip (global, loaded via script tag) for .js_mlmodel unpacking
 *   - Canvas API for image preprocessing (replaces sharp)
 *
 * Pure heatmap and decode logic is imported directly from src/ at build time
 * (esbuild bundles them in — see npm run build:demo).
 */

import {
  maxChannels, threshold, connectedComponents,
  extractOrientedBBoxes, scaleOBBs, sortByReadingOrder,
  findColumnGapFromProfile, splitComponentsAtX,
} from '../src/heatmap.js';

import { buildL2C, greedyCTC, decodeCodec } from '../src/decode.js';

// ort is loaded as a UMD global via <script> tag in index.html.
/* global ort */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// ---------------------------------------------------------------------------
// .js_mlmodel loader (uses JSZip global)
// ---------------------------------------------------------------------------

async function loadJsMlmodel(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status} ${resp.statusText}`);
  const buf = await resp.arrayBuffer();
  const zip = await JSZip.loadAsync(buf);

  const metaStr  = await zip.file('metadata.json').async('string');
  const onnxBuf  = await zip.file('model.onnx').async('arraybuffer');
  const meta     = JSON.parse(metaStr);

  const session  = await ort.InferenceSession.create(onnxBuf, {
    executionProviders: ['wasm'],
  });

  return { session, meta };
}

// ---------------------------------------------------------------------------
// Image preprocessing — Canvas API (mirrors src/preprocess.js)
// ---------------------------------------------------------------------------

/**
 * Draw an HTMLImageElement (or ImageBitmap) onto a canvas scaled to targetH,
 * preserving aspect ratio.
 */
function resizeToHeight(img, targetH) {
  const origW = img.naturalWidth  ?? img.width;
  const origH = img.naturalHeight ?? img.height;
  const scale = targetH / origH;
  const targetW = Math.max(1, Math.round(origW * scale));

  const c = document.createElement('canvas');
  c.width  = targetW;
  c.height = targetH;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, targetW, targetH);
  return c;
}

/**
 * Preprocess a page image for segmentation.
 *
 * Steps (mirrors preprocessPageImage in src/preprocess.js):
 *   1. Resize to model height (proportional width)
 *   2. Read pixels (RGB)
 *   3. Normalize + invert: 1 - pixel/255
 *   4. Reorder HWC → CHW
 *
 * Returns { data: Float32Array (CHW), width, height }
 */
function preprocessPageCanvas(img, meta) {
  const { height: targetH, channels } = meta;
  const c   = resizeToHeight(img, targetH);
  const ctx = c.getContext('2d');
  const { width: W, height: H } = c;

  const raw = ctx.getImageData(0, 0, W, H).data; // RGBA Uint8ClampedArray

  if (channels === 1) {
    // Grayscale: average R,G,B → broadcast to 1 channel, normalize + invert
    const out = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      const r = raw[i * 4], g = raw[i * 4 + 1], b = raw[i * 4 + 2];
      const gray = (r + g + b) / 3;
      out[i] = 1.0 - gray / 255.0;
    }
    // CHW with C=1: layout is just H*W
    return { data: out, width: W, height: H };
  }

  // RGB: normalize + invert, then HWC → CHW
  const hwc = new Float32Array(H * W * 3);
  for (let i = 0; i < H * W; i++) {
    hwc[i * 3    ] = 1.0 - raw[i * 4    ] / 255.0;
    hwc[i * 3 + 1] = 1.0 - raw[i * 4 + 1] / 255.0;
    hwc[i * 3 + 2] = 1.0 - raw[i * 4 + 2] / 255.0;
  }

  // HWC → CHW
  const chw = new Float32Array(3 * H * W);
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      for (let ch = 0; ch < 3; ch++) {
        chw[ch * H * W + h * W + w] = hwc[(h * W + w) * 3 + ch];
      }
    }
  }

  return { data: chw, width: W, height: H };
}

/**
 * Preprocess a line crop (canvas or image element) for recognition.
 *
 * Mirrors preprocessImage in src/preprocess.js.
 * Returns Float32Array in CHW layout.
 */
function preprocessLineCanvas(imgEl, meta) {
  const { height: targetH, channels, pad } = meta;
  const c   = resizeToHeight(imgEl, targetH);
  const W0  = c.width;
  const ctx = c.getContext('2d');
  const raw = ctx.getImageData(0, 0, W0, targetH).data;

  const totalW = W0 + 2 * pad;
  const nPix   = totalW * targetH * channels;

  // Build padded HWC buffer (white = 255)
  const padded = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) padded[i] = 1.0; // pre-invert: white=0 → 1-0/255=1

  if (channels === 1) {
    for (let row = 0; row < targetH; row++) {
      for (let col = 0; col < W0; col++) {
        const r = raw[(row * W0 + col) * 4];
        const g = raw[(row * W0 + col) * 4 + 1];
        const b = raw[(row * W0 + col) * 4 + 2];
        const gray = (r + g + b) / 3;
        padded[row * totalW + (pad + col)] = 1.0 - gray / 255.0;
      }
    }
    // CHW with C=1 is same as H*W
    return { data: padded, width: totalW, height: targetH };
  }

  // RGB
  const paddedRgb = new Uint8Array(totalW * targetH * 3).fill(255);
  for (let row = 0; row < targetH; row++) {
    for (let col = 0; col < W0; col++) {
      const dst = (row * totalW + pad + col) * 3;
      const src = (row * W0 + col) * 4;
      paddedRgb[dst    ] = raw[src    ];
      paddedRgb[dst + 1] = raw[src + 1];
      paddedRgb[dst + 2] = raw[src + 2];
    }
  }

  const nPixFull = totalW * targetH * 3;
  const floatData = new Float32Array(nPixFull);
  for (let i = 0; i < nPixFull; i++) floatData[i] = 1.0 - paddedRgb[i] / 255.0;

  // HWC → CHW
  const chw = new Float32Array(nPixFull);
  for (let h = 0; h < targetH; h++) {
    for (let w = 0; w < totalW; w++) {
      for (let ch = 0; ch < channels; ch++) {
        chw[ch * targetH * totalW + h * totalW + w] = floatData[(h * totalW + w) * channels + ch];
      }
    }
  }
  return { data: chw, width: totalW, height: targetH };
}

// ---------------------------------------------------------------------------
// BrowserSegmenter
// ---------------------------------------------------------------------------

class BrowserSegmenter {
  constructor(session, meta, opts = {}) {
    this._session       = session;
    this._meta          = meta;
    this._noColumnSplit = opts.noColumnSplit || false;
    this._valleyRatio   = opts.valleyRatio   ?? 0.7;
  }

  static async create(url, opts = {}) {
    const { session, meta } = await loadJsMlmodel(url);
    return new BrowserSegmenter(session, meta, opts);
  }

  async segment(imgEl) {
    const origW = imgEl.naturalWidth  ?? imgEl.width;
    const origH = imgEl.naturalHeight ?? imgEl.height;

    const { data: chw, width: W, height: H } = preprocessPageCanvas(imgEl, this._meta);

    const tensor = new ort.Tensor('float32', chw, [1, this._meta.channels, H, W]);
    const output = await this._session.run({ input: tensor });
    const outKey = Object.keys(output)[0];
    const out    = output[outKey];

    const [, C, Hout, Wout] = out.dims;
    const outData = out.data;

    // baseline class indices: all entries under class_mapping.baselines
    const baselineIndices = Object.values(this._meta.class_mapping.baselines);

    const merged  = maxChannels(outData, C, Hout, Wout, baselineIndices);
    const mask    = threshold(merged, 0.5);
    const { labels, count } = connectedComponents(mask, Hout, Wout);

    const colGapX = findColumnGapFromProfile(outData, Hout, Wout, baselineIndices, this._valleyRatio);
    const finalCount = colGapX !== null
      ? splitComponentsAtX(labels, count, Hout, Wout, colGapX)
      : count;

    let obbs = extractOrientedBBoxes(labels, finalCount, Hout, Wout, 20);

    const scaleX = origW / Wout;
    const scaleY = origH / Hout;
    obbs = scaleOBBs(obbs, scaleX, scaleY, origW, origH);

    const imgSplitX = colGapX !== null ? Math.round(colGapX * scaleX) : undefined;

    // Determine type per OBB
    const classMap     = this._meta.class_mapping.baselines;
    const classEntries = Object.entries(classMap);

    const lines = sortByReadingOrder(obbs, origW, origH, this._noColumnSplit, imgSplitX)
      .map(obb => {
        let bestType = classEntries[0][0];
        if (classEntries.length > 1) {
          const dominantIdx = baselineIndices[0];
          bestType = classEntries.find(([, idx]) => idx === dominantIdx)?.[0] ?? classEntries[0][0];
        }
        return { obb, type: bestType };
      });

    return { lines, imageSize: { width: origW, height: origH } };
  }
}

// ---------------------------------------------------------------------------
// BrowserRecognizer
// ---------------------------------------------------------------------------

class BrowserRecognizer {
  constructor(session, meta) {
    this._session = session;
    this._meta    = meta;
    this._l2c     = buildL2C(meta.codec);
  }

  static async create(url) {
    const { session, meta } = await loadJsMlmodel(url);
    return new BrowserRecognizer(session, meta);
  }

  async recognize(imgEl) {
    const { data: chw, width: W, height: H } = preprocessLineCanvas(imgEl, this._meta);

    const tensor = new ort.Tensor('float32', chw, [1, this._meta.channels, H, W]);
    const output = await this._session.run({ input: tensor });
    const outKey = Object.keys(output)[0];
    const out    = output[outKey];

    // output dims: [N, C, W_out] — data is in (C, W) layout for batch 0
    const [, C, Wout] = out.dims;
    const logits     = out.data.subarray(0, C * Wout); // (C, W) layout

    const ctcLabels = greedyCTC(logits, C, Wout);
    const chars     = decodeCodec(ctcLabels, this._l2c);

    const inputWidth = W;
    const scaledChars = chars.map(ch => ({
      char: ch.char,
      conf: ch.conf,
      x0: Math.round(ch.t0 * inputWidth / Wout),
      x1: Math.round(ch.t1 * inputWidth / Wout),
    }));

    return { text: scaledChars.map(c => c.char).join(''), chars: scaledChars };
  }
}

// ---------------------------------------------------------------------------
// Line crop extraction (mirrors pipeline.js extractLineCrop using Canvas)
// ---------------------------------------------------------------------------

function estimateLineHeight(lines, imageSize) {
  if (lines.length < 2) return Math.round(imageSize.height / 30);
  const maxGap = imageSize.height * 0.3;
  const gaps = [];
  for (let i = 1; i < lines.length; i++) {
    const g = lines[i].obb.cy - lines[i - 1].obb.cy;
    if (g > 2 && g < maxGap) gaps.push(g);
  }
  if (gaps.length === 0) return Math.round(imageSize.height / 30);
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

function extractLineCropCanvas(source, obb, origW, origH, lineHeight, topline, opts = {}) {
  const { cx, cy, angle, w: obbW } = obb;
  const upRatio    = opts.expandUp   ?? (topline ? 0.35 : 0.85);
  const downRatio  = opts.expandDown ?? (topline ? 0.85 : 0.35);
  const expandUp   = lineHeight * upRatio;
  const expandDown = lineHeight * downRatio;
  const hPad = Math.ceil(lineHeight * 0.1);
  const hw   = obbW / 2 + hPad;

  const finalW = Math.max(1, Math.ceil(2 * hw));
  const finalH = Math.max(1, Math.ceil(expandUp + expandDown));

  // Compute the rotated crop corners for the overlay
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const vx = sinA, vy = -cosA; // perpendicular "above baseline" direction
  const rotCorners = [
    [cx - hw * cosA + expandUp   * vx, cy - hw * sinA + expandUp   * vy],
    [cx + hw * cosA + expandUp   * vx, cy + hw * sinA + expandUp   * vy],
    [cx + hw * cosA - expandDown * vx, cy + hw * sinA - expandDown * vy],
    [cx - hw * cosA - expandDown * vx, cy - hw * sinA - expandDown * vy],
  ];
  const rxs = rotCorners.map(c => c[0]), rys = rotCorners.map(c => c[1]);
  extractLineCropCanvas._lastBounds = {
    left:       Math.max(0, Math.floor(Math.min(...rxs))),
    top:        Math.max(0, Math.floor(Math.min(...rys))),
    right:      Math.min(origW - 1, Math.ceil(Math.max(...rxs))),
    bottom:     Math.min(origH - 1, Math.ceil(Math.max(...rys))),
    rotCorners,
  };

  const c   = document.createElement('canvas');
  c.width   = finalW;
  c.height  = finalH;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, finalW, finalH);

  const { left, top, right, bottom } = extractLineCropCanvas._lastBounds;
  if (Math.abs(angle * 180 / Math.PI) < 0.5) {
    ctx.drawImage(source, left, top, right - left, bottom - top, 0, 0, finalW, finalH);
  } else {
    // Translate so (cx, cy) lands at the baseline anchor, rotate by -angle,
    // draw the full source — the text line emerges horizontal.
    ctx.save();
    ctx.translate(finalW / 2, expandUp);
    ctx.rotate(-angle);
    ctx.drawImage(source, -cx, -cy);
    ctx.restore();
  }

  return c;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run the full segmentation + recognition pipeline on an image element.
 *
 * @param {HTMLImageElement} imgEl
 * @param {string}           segUrl   URL of segmentation .js_mlmodel
 * @param {string}           recUrl   URL of recognition .js_mlmodel
 * @param {{ onStatus?: (msg:string)=>void, onLine?: (line:object)=>void }} opts
 */
export async function runPipeline(imgEl, segUrl, recUrl, opts = {}) {
  const { onStatus = () => {}, onLine = () => {}, noColumnSplit = false,
          expandUp, expandDown } = opts;

  onStatus('Loading segmentation model…');
  const segmenter = await BrowserSegmenter.create(segUrl, { noColumnSplit });

  onStatus('Loading recognition model…');
  const recognizer = await BrowserRecognizer.create(recUrl);

  onStatus('Segmenting page…');
  const { lines, imageSize } = await segmenter.segment(imgEl);

  if (lines.length === 0) {
    onStatus('No lines detected.');
    return [];
  }

  const lineHeight = estimateLineHeight(lines, imageSize);
  const topline    = segmenter._meta.topline || false;

  const yield_ = () => new Promise(r => setTimeout(r, 0));

  const results = [];
  for (let i = 0; i < lines.length; i++) {
    const { obb, type } = lines[i];
    onStatus(`Recognizing line ${i + 1} / ${lines.length}…`);
    await yield_(); // release the main thread so the browser can repaint

    const cropCanvas = extractLineCropCanvas(
      imgEl, obb,
      imageSize.width, imageSize.height,
      lineHeight, topline,
      { expandUp, expandDown }
    );
    const cropBounds = { ...extractLineCropCanvas._lastBounds };

    const { text, chars } = await recognizer.recognize(cropCanvas);
    const result = { obb, type, text, chars, cropBounds };
    results.push(result);
    onLine(result);
  }

  return results;
}
