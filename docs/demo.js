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

import { selectDetections, invertMapping, toRegion } from '../src/detections.js';

// ort is loaded as a UMD global via <script> tag in index.html.
/* global ort */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

// Run the WASM backend in ONNX Runtime's own worker. Without this, compiling the
// model and every session.run() happen on the page's thread — a full-page
// segmentation pass takes seconds and the UI cannot repaint for its duration.
ort.env.wasm.proxy = true;

// Multi-threaded WASM only exists on a cross-origin isolated page (COOP/COEP).
// serve-demo.js sends those headers; GitHub Pages cannot, and there ORT stays
// single-threaded — slower, but off the main thread either way.
export const DEFAULT_THREADS = 4;

/** Whether this page can run ONNX Runtime with more than one thread. */
export const threadsAvailable = Boolean(globalThis.crossOriginIsolated);

/** Most threads worth offering: every logical core the browser reports. */
export const maxThreads = navigator.hardwareConcurrency || 1;

// ORT reads numThreads once, when it initialises WASM for the first session;
// later changes are ignored until the page reloads.
let _wasmInitialised = false;

/**
 * Set the number of inference threads. Returns false once a model is loaded,
 * since the value can then only take effect after a reload.
 */
export function setThreads(n) {
  if (_wasmInitialised) return false;
  ort.env.wasm.numThreads = threadsAvailable ? Math.max(1, Math.min(n, maxThreads)) : 1;
  return true;
}

setThreads(DEFAULT_THREADS);

// ---------------------------------------------------------------------------
// .js_mlmodel loader (uses JSZip global)
// ---------------------------------------------------------------------------

/**
 * Sessions already built, keyed by model URL. Models are tens of megabytes and
 * compiling one costs seconds, so a session is created once per page load and
 * reused across runs and model switches. Promises are cached rather than
 * resolved values, so concurrent loads of the same URL share one download.
 *
 * @type {Map<string, Promise<{session: ort.InferenceSession, meta: object}>>}
 */
const _sessionCache = new Map();

/** Whether a model is already loaded, so callers can skip the loading status. */
export function isModelCached(url) {
  return _sessionCache.has(url);
}

/**
 * Fetch a .js_mlmodel and build an inference session, reporting download progress.
 *
 * @param {string} url
 * @param {{ onProgress?: (fraction: number|null) => void }} [opts]
 *        onProgress receives 0..1, or null when the size is unknown.
 */
function loadJsMlmodel(url, opts = {}) {
  const cached = _sessionCache.get(url);
  if (cached) return cached;

  const pending = _loadJsMlmodel(url, opts).catch(err => {
    // A failed load must not poison the cache — let the next run retry.
    _sessionCache.delete(url);
    throw err;
  });
  _sessionCache.set(url, pending);
  return pending;
}

async function _loadJsMlmodel(url, { onProgress } = {}) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status} ${resp.statusText}`);

  const buf = await readBodyWithProgress(resp, onProgress);
  const zip = await JSZip.loadAsync(buf);

  const metaStr  = await zip.file('metadata.json').async('string');
  const onnxBuf  = await zip.file('model.onnx').async('arraybuffer');
  const meta     = JSON.parse(metaStr);

  const session  = await createSession(onnxBuf);

  return { session, meta };
}

/**
 * Read a response body, reporting progress against Content-Length.
 * Falls back to a plain arrayBuffer() when streaming or the length is unavailable.
 */
async function readBodyWithProgress(resp, onProgress) {
  const total = Number(resp.headers.get('content-length')) || 0;
  if (!onProgress || !resp.body || !total) {
    if (onProgress) onProgress(null);
    return resp.arrayBuffer();
  }

  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / total));
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

/**
 * Build a session, falling back to in-page execution if the proxy worker fails.
 *
 * The fallback matters because a broken proxy would otherwise take the whole demo
 * down; running on the main thread is slow and blocking, but it works.
 */
async function createSession(onnxBuf) {
  _wasmInitialised = true;
  try {
    return await ort.InferenceSession.create(onnxBuf, { executionProviders: ['wasm'] });
  } catch (err) {
    if (!ort.env.wasm.proxy) throw err;
    console.warn('ONNX Runtime proxy worker unavailable, falling back to the main ' +
                 'thread (the page will block during inference):', err);
    ort.env.wasm.proxy = false;
    return ort.InferenceSession.create(onnxBuf, { executionProviders: ['wasm'] });
  }
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

  // RGB: normalize + invert straight into CHW. Going via an intermediate HWC
  // array would mean a second pass and a second ~28 MB allocation on a full page.
  const plane = H * W;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const src = i * 4;
    chw[i            ] = 1.0 - raw[src    ] / 255.0;
    chw[i + plane    ] = 1.0 - raw[src + 1] / 255.0;
    chw[i + 2 * plane] = 1.0 - raw[src + 2] / 255.0;
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
    const { session, meta } = await loadJsMlmodel(url, { onProgress: opts.onProgress });
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
    const mask    = threshold(merged, Hout, Wout, 0.5);
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
// BrowserZoneDetector — D-FINE layout regions (mirrors src/dfine.js)
// ---------------------------------------------------------------------------

/**
 * Preprocess a page for D-FINE: stretch to the model's fixed input size and
 * scale to 0..1, without inversion (mirrors preprocessDetectionImage).
 */
function preprocessDetectionCanvas(img, meta) {
  const [H, W] = meta.image_size;
  const c = document.createElement('canvas');
  c.width  = W;
  c.height = H;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, W, H);
  const raw = ctx.getImageData(0, 0, W, H).data;

  const plane = H * W;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i            ] = raw[i * 4    ] / 255;
    chw[i + plane    ] = raw[i * 4 + 1] / 255;
    chw[i + 2 * plane] = raw[i * 4 + 2] / 255;
  }
  return { data: chw, width: W, height: H };
}

class BrowserZoneDetector {
  constructor(session, meta, opts = {}) {
    this._session        = session;
    this._meta           = meta;
    this._scoreThreshold = opts.scoreThreshold ?? 0.5;
    this._regionMap      = invertMapping(meta.class_mapping?.regions);
  }

  static async create(url, opts = {}) {
    const { session, meta } = await loadJsMlmodel(url, { onProgress: opts.onProgress });
    if (meta.architecture !== 'dfine') {
      throw new Error(`zone model is not a D-FINE model (architecture: ${meta.architecture})`);
    }
    return new BrowserZoneDetector(session, meta, opts);
  }

  /** Detect layout regions; returns [{ bbox, polygon, type, score }], top-to-bottom. */
  async detect(imgEl) {
    const origW = imgEl.naturalWidth  ?? imgEl.width;
    const origH = imgEl.naturalHeight ?? imgEl.height;

    const { data, width, height } = preprocessDetectionCanvas(imgEl, this._meta);
    const input = new ort.Tensor('float32', data, [1, 3, height, width]);
    const { scores, boxes } = await this._session.run({ input });

    const detections = selectDetections(scores.data, boxes.data, scores.dims[1], scores.dims[2], {
      topK: this._meta.num_top_queries ?? 300,
      scoreThreshold: this._scoreThreshold,
      width: origW,
      height: origH,
    });

    const zones = detections
      .filter(det => this._regionMap.has(det.label))
      .map(det => toRegion(det, this._regionMap.get(det.label)));
    zones.sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
    return zones;
  }
}

/** Index of the smallest zone containing (x, y), or -1. */
function zoneAt(zones, x, y) {
  let best = -1, bestArea = Infinity;
  zones.forEach(({ bbox: [x0, y0, x1, y1] }, i) => {
    const area = (x1 - x0) * (y1 - y0);
    if (x > x0 && x < x1 && y > y0 && y < y1 && area < bestArea) {
      best = i;
      bestArea = area;
    }
  });
  return best;
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

  static async create(url, opts = {}) {
    const { session, meta } = await loadJsMlmodel(url, { onProgress: opts.onProgress });
    return new BrowserRecognizer(session, meta);
  }

  async recognize(imgEl) {
    const { data: chw, width: W, height: H } = preprocessLineCanvas(imgEl, this._meta);

    const tensor = new ort.Tensor('float32', chw, [1, this._meta.channels, H, W]);
    const feeds  = { input: tensor };
    // PP-OCRv6 graphs take the unpadded width as a second input so their
    // attention neck can mask the padding out (see src/recognizer.js).
    if (this._meta.seq_lens_input || this._session.inputNames.includes('seq_lens')) {
      feeds.seq_lens = new ort.Tensor('int64', BigInt64Array.from([BigInt(W)]), [1]);
    }
    const output = await this._session.run(feeds);
    const out    = output['output'] ?? output[Object.keys(output)[0]];

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
 * Build an onProgress callback that reports model loading through onStatus.
 * Cached models report once and never show a percentage.
 */
function loadReporter(name, url, onStatus) {
  if (isModelCached(url)) {
    onStatus(`Using cached ${name} model…`);
    return undefined;
  }
  onStatus(`Loading ${name} model…`);
  return fraction => {
    onStatus(fraction === null
      ? `Loading ${name} model…`
      : `Loading ${name} model… ${Math.round(fraction * 100)}%`);
  };
}

/**
 * Run the full segmentation + recognition pipeline on an image element.
 *
 * @param {HTMLImageElement} imgEl
 * @param {string}           segUrl   URL of segmentation .js_mlmodel
 * @param {string}           recUrl   URL of recognition .js_mlmodel
 * @param {{ onStatus?: (msg:string)=>void, onLine?: (line:object)=>void,
 *           zoneUrl?: string, onZones?: (zones:object[])=>void }} opts
 *        zoneUrl: D-FINE .js_mlmodel for layout zones; each line then carries
 *        `zone`, the index of the smallest zone containing its centre (or -1).
 */
export async function runPipeline(imgEl, segUrl, recUrl, opts = {}) {
  const { onStatus = () => {}, onLine = () => {}, onZones = () => {},
          noColumnSplit = false, zoneUrl, expandUp, expandDown } = opts;

  const zoneDetector = zoneUrl && await BrowserZoneDetector.create(zoneUrl, {
    onProgress: loadReporter('zone', zoneUrl, onStatus),
  });

  const segmenter = await BrowserSegmenter.create(segUrl, {
    noColumnSplit,
    onProgress: loadReporter('segmentation', segUrl, onStatus),
  });

  const recognizer = await BrowserRecognizer.create(recUrl, {
    onProgress: loadReporter('recognition', recUrl, onStatus),
  });

  let zones = [];
  if (zoneDetector) {
    onStatus('Detecting zones…');
    zones = await zoneDetector.detect(imgEl);
    onZones(zones);
  }

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
    const zone = zoneAt(zones, obb.cx, obb.cy);
    const result = { obb, type, text, chars, cropBounds, zone };
    results.push(result);
    onLine(result);
  }

  return results;
}
